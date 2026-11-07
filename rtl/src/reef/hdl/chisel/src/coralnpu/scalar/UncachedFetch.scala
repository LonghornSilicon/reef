// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package coralnpu

import chisel3._
import chisel3.util._
import common._

class PredecodeOutput(p: Parameters) extends Bundle {
    val insts = Vec(p.fetchInstrSlots, new FetchInstruction(p))
    val count = UInt(4.W)
    val nextPc = UInt(p.instructionBits.W)
}

class FetchResponse(p: Parameters) extends Bundle {
    val addr = UInt(p.fetchAddrBits.W)
    val inst = Vec(p.fetchInstrSlots, UInt(p.instructionBits.W))
    val fault = Bool()
}

class Instruction(p: Parameters) extends Bundle {
    val addr = UInt(p.fetchAddrBits.W)
    val inst = UInt(p.instructionBits.W)
}

// TODO(atv): Privatize this and FetchControl
// Module which is responsible for performing
// memory fetches which are requested by
// `FetchControl`.
// `ibus` should be treated like Chisel's
// `IrrevocableIO`.
class Fetcher(p: Parameters) extends Module {
  val io = IO(new Bundle {
    val ctrl = Flipped(Decoupled(UInt(p.fetchAddrBits.W)))
    val fetch = Output(Valid(new FetchResponse(p)))
    val ibus = new IBusIO(p)
  })

  val lsb = log2Ceil(p.fetchDataBits / 8)
  assert((p.fetchDataBits == 128 && lsb == 4) || (p.fetchDataBits == 256 && lsb == 5))

  // The actual fetch transaction goes through without stopping.
  io.ibus.valid := io.ctrl.valid
  io.ibus.addr := Cat(io.ctrl.bits(p.fetchAddrBits - 1, lsb), 0.U(lsb.W))
  io.ctrl.ready := io.ibus.ready

  // Address is buffered to accompany results.
  val ibusFired = io.ctrl.valid && io.ibus.ready
  val ibusCmd = RegNext(ForceZero(MakeValid(ibusFired, io.ctrl.bits)), MakeInvalid(UInt(32.W)))
  // Fault is also buffered to accompany results.
  val fault = RegNext(io.ibus.fault.valid, false.B)
  io.fetch.valid := ibusCmd.valid
  io.fetch.bits.addr := ibusCmd.bits
  io.fetch.bits.fault := fault
  for (i <- 0 until p.fetchInstrSlots) {
    val offset = p.instructionBits * i
    io.fetch.bits.inst(i) := io.ibus.rdata(offset + p.instructionBits - 1, offset)
  }
}

class FetchControl(p: Parameters) extends Module {
    val io = IO(new Bundle {
        val fetchFault = Valid(UInt(32.W))
        val csr = new CsrInIO(p)
        val iflush = Input(Valid(UInt(32.W)))
        val branch = Input(Valid(UInt(p.fetchAddrBits.W)))
        val fetchData = Input(Valid(new FetchResponse(p)))
        val linkPort = Flipped(new RegfileLinkPortIO)

        val fetchAddr = Decoupled(UInt(p.fetchAddrBits.W))
        val bufferRequest = DecoupledVectorIO(new FetchInstruction(p), p.fetchInstrSlots)
        val bufferSpaces = Input(UInt(log2Ceil(p.fetchInstrSlots * 2 + 1).W))
    })

    def PredictJump(addr: UInt, inst: UInt): ValidIO[UInt] = {
      assert(p.instructionBits == 32)
      val jal = inst === BitPat("b????????????????????_?????_1101111")
      val immjal = Cat(Fill(12, inst(31)), inst(19,12), inst(20), inst(30,21), 0.U(1.W))
      val bxx = inst === BitPat("b???????_?????_?????_???_?????_1100011") &&
                  inst(31) && inst(14,13) =/= 1.U
      val immbxx = Cat(Fill(20, inst(31)), inst(7), inst(30,25), inst(11,8), 0.U(1.W))
      val immed = Mux(inst(2), immjal, immbxx)

      val valid = jal || bxx
      val target = addr + immed

      MakeValid(valid, target)
    }

    def Predecode(fetchResponse: FetchResponse): PredecodeOutput = {
      val addr = fetchResponse.addr
      val lsb = log2Ceil(p.fetchDataBits / 8)
      assert((p.fetchDataBits == 128 && lsb == 4) || (p.fetchDataBits == 256 && lsb == 5))
      val baseAddr = addr(p.fetchAddrBits - 1, lsb)
      val startElem = addr(lsb - 1, lsb - log2Ceil(p.fetchInstrSlots))

      val insts = ShiftVectorRight(fetchResponse.inst, startElem)
      val addrs = VecInit.tabulate(p.fetchInstrSlots)(i =>
          addr + (i * 4).U
      )

      val branchTargets = VecInit.tabulate(p.fetchInstrSlots)(i =>
          PredictJump(addrs(i), insts(i))
      )

      val validsIn = VecInit.tabulate(p.fetchInstrSlots)(i =>
          i.U < p.fetchInstrSlots.U - startElem
      )
      val jumped = VecInit.tabulate(p.fetchInstrSlots)(i =>
          validsIn(i) && branchTargets(i).valid
      )
      val firstJumpOH = VecInit(PriorityEncoderOH(jumped))

      // Have we jumped before the instruction i
      val hasJumpedBefore = VecInit(jumped.scan(false.B)(_||_).take(p.fetchInstrSlots))

      val validsOut = VecInit.tabulate(p.fetchInstrSlots)(i =>
          validsIn(i) && !hasJumpedBefore(i)
      )

      val nextFetchPc = MuxUpTo1H(Cat(baseAddr + 1.U, 0.U(lsb.W)),
          (0 until p.fetchInstrSlots).map(i => firstJumpOH(i) -> branchTargets(i).bits))

      val result = MakeWireBundle[PredecodeOutput](
          new PredecodeOutput(p),
          _.insts -> VecInit.tabulate(p.fetchInstrSlots)(i =>
              MakeWireBundle[FetchInstruction](
                  new FetchInstruction(p),
                  _.addr -> addrs(i),
                  _.inst -> insts(i),
                  _.brchFwd -> jumped(i),
              )
          ),
          _.count -> validsOut.count(x => x),
          _.nextPc -> nextFetchPc,
      )

      result
    }

    val predecode = Predecode(io.fetchData.bits)

    io.bufferRequest.bits := predecode.insts

    val pastBranchOrFlush = RegInit(false.B)
    val currentBranchOrFlush = io.iflush.valid || io.branch.valid
    val ongoingBranchOrFlush = pastBranchOrFlush || currentBranchOrFlush

    // If we have faulted we should stop making any new attempts until a branch resolves it.
    val faulted = RegInit(false.B)
    val fetchFaultValid = (faulted || (io.fetchData.valid && io.fetchData.bits.fault)) &&
        !io.branch.valid
    io.fetchFault := MakeValid(fetchFaultValid, io.fetchData.bits.addr)
    faulted := fetchFaultValid

    // Send out results. All branch or flush, current or past, will make us
    // discard results.
    // TODO(davidgao): ForceZero it when invalid?
    val writeToBuffer = io.fetchData.valid && !fetchFaultValid && !ongoingBranchOrFlush
    val nValid = Mux(writeToBuffer, predecode.count, 0.U)
    io.bufferRequest.nValid := nValid

    // PC is initialized with the CSR value below upon leaving reset.
    val pc = RegInit(MakeInvalid(UInt(32.W)))
    val pcNext = MuxCase(pc.bits, Seq(
        (!pc.valid) -> Cat(io.csr.value(0)(31,2), 0.U(2.W)),  // We're leaving reset.
        io.iflush.valid -> io.iflush.bits,
        io.branch.valid -> io.branch.bits,
        writeToBuffer -> predecode.nextPc,
        // At this point `io.fetchData.valid` is false. We did not fire a
        // transaction last cycle. This could be a delay in results or a block
        // on our side. EAGAIN.
    ))
    // PC will always be valid as soon as we leave reset.
    pc := MakeValid(pcNext)

    // Buffer space for the fetched instructions are guaranteed upon initiation
    // of the transaction. We can only start a new fetch if there is sufficient
    // space AFTER we push what we have on hand.
    val insufficientBuffer = io.bufferSpaces < nValid +& p.fetchInstrSlots.U
    // Past branch or flush doesn't block us from initiating new fetches.
    val blockNewFetch = !pc.valid ||  // We're stil in reset.
                        io.fetchData.valid || // Wait one cycle for next fetch.
                        currentBranchOrFlush ||
                        insufficientBuffer ||
                        fetchFaultValid
    val fetch = ForceZero(MakeValid(!blockNewFetch, pc.bits))

    // All branch or flush are cleared once we're able to initiate a new fetch.
    pastBranchOrFlush := ongoingBranchOrFlush && blockNewFetch

    io.fetchAddr.valid := fetch.valid
    io.fetchAddr.bits := fetch.bits
}

class UncachedFetch(p: Parameters) extends FetchUnit(p) {
  // TODO(derekjchow): Make Bru use valid interface
  val branch = MuxCase(
      MakeInvalid(UInt(p.fetchAddrBits.W)),
      (0 until p.instructionLanes).map(i =>
          io.branch(i).valid -> MakeValid(io.branch(i).value)
      ))

  val ctrl = Module(new FetchControl(p))
  ctrl.io.csr <> io.csr
  ctrl.io.branch := branch
  val debug_iflush = Seq(
    io.debug_pc.valid -> MakeValid(io.debug_pc.bits),
  )
  ctrl.io.iflush := MuxCase(MakeInvalid(UInt(p.fetchAddrBits.W)), Seq(
    io.iflush.valid -> MakeValid(io.iflush.pcNext),
  ) ++ debug_iflush)
  ctrl.io.linkPort := io.linkPort
  // TODO(derekjchow): Maybe do something with back pressure?
  io.iflush.ready := true.B

  val fetcher = Module(new Fetcher(p))
  fetcher.io.ctrl <> ctrl.io.fetchAddr
  ctrl.io.fetchData := fetcher.io.fetch
  fetcher.io.ibus <> io.ibus

  val window = p.fetchInstrSlots * 2
  val instructionBuffer = Module(new InstructionBuffer(
      new FetchInstruction(p), p.fetchInstrSlots, window))
  instructionBuffer.io.feedIn <> ctrl.io.bufferRequest
  io.inst.lanes <> instructionBuffer.io.out.take(4)
  instructionBuffer.io.flush := io.iflush.valid || branch.valid || io.debug_pc.valid
  ctrl.io.bufferSpaces := instructionBuffer.io.nSpace

  val pc = RegInit(0.U(p.fetchAddrBits.W))
  pc := Mux(instructionBuffer.io.out(0).valid, instructionBuffer.io.out(0).bits.addr, pc)
  io.pc := pc
  io.fault := ctrl.io.fetchFault
}
