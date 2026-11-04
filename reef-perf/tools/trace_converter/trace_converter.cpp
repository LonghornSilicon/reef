//===- trace_converter.cpp - reef_perf trace to Kanata log ----------------===//
///
/// \file
/// Converts the instruction timing trace written by `reef_perf --trace` into a
/// Kanata log (format 0004), which pipeline viewers such as Konata open.
///
///   trace_converter <trace.tsv> <out.log> [<start_pc> <end_pc>]
///
/// With a PC range, the log starts at the first event of an instruction at
/// <start_pc>. It stops once an instruction at <end_pc> has been seen and every
/// instruction in flight has retired. PCs are decimal or 0x-prefixed hex.
///
//===----------------------------------------------------------------------===//

#include "common/Types.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace reef_perf;

namespace {

/// One line of the trace: cycle, activity, seq, pc and disassembly.
struct TraceEvent {
  Cycle At = 0;
  std::string Activity;
  std::string Seq;
  Address PC = 0;
  std::string Disasm;
};

/// Writes trace events as a Kanata log.
class KanataWriter {
public:
  explicit KanataWriter(std::ostream &OS) : OS(OS) { OS << "Kanata\t0004\n"; }

  /// Writes \p Event. If \p OnlyKnown, skips events of instructions that have
  /// not appeared before.
  void write(const TraceEvent &Event, bool OnlyKnown);

  /// \returns true while an instruction has been seen but has not retired.
  bool hasInFlight() const { return InFlight != 0; }

  /// \returns the number of instructions written so far.
  size_t getNumInsts() const { return Ids.size(); }

private:
  void advanceTo(Cycle At);

  std::ostream &OS;
  std::unordered_map<std::string, size_t> Ids; // Seq -> Kanata instruction id.
  size_t InFlight = 0;
  uint64_t NextRetireId = 0;
  bool HasCycle = false;
  Cycle CurrentCycle = 0;
};

} // namespace

/// \returns the stage name the viewer shows for \p Activity.
static std::string getStageName(const std::string &Activity) {
  static const std::unordered_map<std::string, std::string> Names = {
      {"fetch", "F"},
      {"dispatch", "D"},
      {"result_ready", "RR"},
      {"complete", "C"},
      {"ScalarExec_Buffer", "SCb"},
      {"ScalarExec", "SC"},
      {"LSU_Buffer", "LSUb"},
      {"LSU", "LSU"},
      {"Vector_Buffer", "Vb"},
      {"Vector", "V"},
      {"Matrix_Buffer", "Mb"},
      {"Matrix", "M"},
  };
  auto It = Names.find(Activity);
  return It == Names.end() ? Activity : It->second;
}

void KanataWriter::advanceTo(Cycle At) {
  if (!HasCycle) {
    OS << "C=\t" << At << "\n";
    HasCycle = true;
  } else if (At != CurrentCycle) {
    OS << "C\t" << At - CurrentCycle << "\n";
  }
  CurrentCycle = At;
}

void KanataWriter::write(const TraceEvent &Event, bool OnlyKnown) {
  auto It = Ids.find(Event.Seq);
  if (OnlyKnown && It == Ids.end())
    return;
  advanceTo(Event.At);

  if (It == Ids.end()) {
    It = Ids.emplace(Event.Seq, Ids.size()).first;
    ++InFlight;
    OS << "I\t" << It->second << "\t0\t0\n";
    OS << "L\t" << It->second << "\t0\t" << Event.Disasm << "\n";
  }
  if (Event.Activity == "retire") {
    OS << "R\t" << It->second << "\t" << NextRetireId++ << "\t0\n";
    --InFlight;
    return;
  }
  OS << "S\t" << It->second << "\t0\t" << getStageName(Event.Activity) << "\n";
}

/// Parses one tab-separated trace line. \throws std::runtime_error if it does
/// not have the five columns of FuncSim::writeTrace.
static TraceEvent parseEvent(const std::string &Line) {
  std::vector<std::string> Fields;
  size_t Begin = 0;
  for (int Col = 0; Col < 4; ++Col) {
    size_t Tab = Line.find('\t', Begin);
    if (Tab == std::string::npos)
      throw std::runtime_error("malformed trace line: " + Line);
    Fields.push_back(Line.substr(Begin, Tab - Begin));
    Begin = Tab + 1;
  }
  Fields.push_back(Line.substr(Begin));

  TraceEvent Event;
  Event.At = std::stoull(Fields[0]);
  Event.Activity = Fields[1];
  Event.Seq = Fields[2];
  Event.PC = static_cast<Address>(std::stoull(Fields[3], nullptr, 16));
  Event.Disasm = Fields[4];
  return Event;
}

/// Converts the trace named on the command line. \returns the exit code.
static int run(int Argc, char **Argv) {
  if (Argc != 3 && Argc != 5) {
    std::cerr << "usage: " << Argv[0]
              << " <trace.tsv> <out.log> [<start_pc> <end_pc>]\n";
    return 2;
  }
  bool Bounded = Argc == 5;
  Address Start = 0;
  Address End = 0;
  if (Bounded) {
    Start = static_cast<Address>(std::stoull(Argv[3], nullptr, 0));
    End = static_cast<Address>(std::stoull(Argv[4], nullptr, 0));
  }

  std::ifstream In(Argv[1]);
  if (!In)
    throw std::runtime_error(std::string("cannot open ") + Argv[1]);
  std::ofstream Out(Argv[2]);
  if (!Out)
    throw std::runtime_error(std::string("cannot open ") + Argv[2]);

  KanataWriter Writer(Out);
  bool ReachedStart = !Bounded;
  bool ReachedEnd = false;
  std::string Line;
  std::getline(In, Line); // Header.
  while (std::getline(In, Line)) {
    TraceEvent Event = parseEvent(Line);
    if (Bounded && Event.PC == Start)
      ReachedStart = true;
    if (Bounded && Event.PC == End)
      ReachedEnd = true;
    bool Started = Writer.getNumInsts() != 0;
    if (ReachedEnd && Started && !Writer.hasInFlight())
      break;
    if (ReachedStart)
      Writer.write(Event, ReachedEnd && Started);
  }

  Out.close();
  if (!Out)
    throw std::runtime_error(std::string("failed to write ") + Argv[2]);
  std::cout << "Wrote " << Argv[2] << " (" << Writer.getNumInsts()
            << " instructions)\n";
  return 0;
}

int main(int argc, char **argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception &E) {
    std::cerr << "trace_converter: " << E.what() << "\n";
    return 1;
  }
}
