//===- trace_converter.cpp - reef_perf trace to Kanata log ----------------===//
///
/// \file
/// Converts the instruction timing trace written by `reef_perf --trace` into a
/// Kanata log (format 0004), which pipeline viewers such as Konata open.
///
///   trace_converter <trace.tsv> <out.log> [<start_seq> <end_seq>]
///
/// With a sequence range, the log includes instructions from <start_seq>
/// through <end_seq>, inclusive. Sequence numbers are decimal.
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
  uint64_t Seq = 0;
  Address PC = 0;
  std::string Disasm;
};

/// Writes trace events as a Kanata log.
class KanataWriter {
public:
  explicit KanataWriter(std::ostream &OS) : OS(OS) { OS << "Kanata\t0004\n"; }

  /// Writes \p Event.
  void write(const TraceEvent &Event);

  /// \returns the number of instructions written so far.
  size_t getNumInsts() const { return Ids.size(); }

private:
  void advanceTo(Cycle At);

  std::ostream &OS;
  std::unordered_map<uint64_t, size_t> Ids; // Seq -> Kanata instruction id.
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

void KanataWriter::write(const TraceEvent &Event) {
  auto It = Ids.find(Event.Seq);
  advanceTo(Event.At);

  if (It == Ids.end()) {
    It = Ids.emplace(Event.Seq, Ids.size()).first;
    OS << "I\t" << It->second << "\t0\t0\n";
    OS << "L\t" << It->second << "\t0\t" << Event.Disasm << "\n";
  }
  if (Event.Activity == "retire") {
    OS << "R\t" << It->second << "\t" << NextRetireId++ << "\t0\n";
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
  Event.Seq = std::stoull(Fields[2], nullptr, 10);
  Event.PC = static_cast<Address>(std::stoull(Fields[3], nullptr, 16));
  Event.Disasm = Fields[4];
  return Event;
}

/// Converts the trace named on the command line. \returns the exit code.
static int run(int Argc, char **Argv) {
  if (Argc != 3 && Argc != 5) {
    std::cerr << "usage: " << Argv[0]
              << " <trace.tsv> <out.log> [<start_seq> <end_seq>]\n";
    return 2;
  }
  bool Bounded = Argc == 5;
  uint64_t Start = 0;
  uint64_t End = 0;
  if (Bounded) {
    Start = std::stoull(Argv[3], nullptr, 10);
    End = std::stoull(Argv[4], nullptr, 10);
    if (Start > End)
      throw std::runtime_error("start sequence must be less than or equal to "
                               "end sequence");
  }

  std::ifstream In(Argv[1]);
  if (!In)
    throw std::runtime_error(std::string("cannot open ") + Argv[1]);
  std::ofstream Out(Argv[2]);
  if (!Out)
    throw std::runtime_error(std::string("cannot open ") + Argv[2]);

  KanataWriter Writer(Out);
  std::string Line;
  std::getline(In, Line); // Header.
  while (std::getline(In, Line)) {
    TraceEvent Event = parseEvent(Line);
    if (Bounded && (Event.Seq < Start || Event.Seq > End))
      continue;
    Writer.write(Event);
    if (Bounded && Event.Seq == End && Event.Activity == "retire")
      break;
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
