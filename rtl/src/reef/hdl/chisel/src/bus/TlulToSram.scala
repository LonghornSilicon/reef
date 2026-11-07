// Copyright 2026 Google LLC
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

package bus

import chisel3._
import chisel3.util._
import coralnpu.Parameters

class Sram128IO(val addrWidth: Int) extends Bundle {
  val enable = Input(Bool())
  val write  = Input(Bool())
  val addr   = Input(UInt(addrWidth.W))
  val wdata  = Input(UInt(128.W))
  val wmask  = Input(UInt(16.W))
  val rdata  = Output(UInt(128.W))
  val rvalid = Output(Bool())
}

class TlulToSram(p: Parameters, sramAddressWidth: Int) extends Module {
  val tlul_p = new TLULParameters(p)
  val io = IO(new Bundle {
    val tl   = Flipped(new OpenTitanTileLink.Host2Device(tlul_p))
    val sram = Flipped(new Sram128IO(sramAddressWidth))
  })

  // Explicit Queues for A and D Channels to decouple bus and memory latency
  val a_q = Module(new Queue(chiselTypeOf(io.tl.a.bits), 2))
  val d_q = Module(new Queue(chiselTypeOf(io.tl.d.bits), 2))

  io.tl.a <> a_q.io.enq
  io.tl.d <> d_q.io.deq

  // Metadata Queue to track outstanding requests through the SRAM pipeline
  val metadata_q = Module(
    new Queue(
      new Bundle {
        val source = UInt(tlul_p.o.W)
        val size   = UInt(tlul_p.z.W)
        val opcode = UInt(3.W)
      },
      4
    )
  )

  // Internal Request Logic
  // Only issue a request to SRAM if we have a pending TL-UL request AND
  // space to store its metadata and eventual response.
  val can_issue = a_q.io.deq.valid && d_q.io.enq.ready && metadata_q.io.enq.ready
  a_q.io.deq.ready := can_issue

  io.sram.enable := can_issue
  io.sram.write  := a_q.io.deq.bits.opcode === TLULOpcodesA.PutFullData.asUInt ||
    a_q.io.deq.bits.opcode === TLULOpcodesA.PutPartialData.asUInt
  // SRAM is word-addressed (128-bit / 16-byte words)
  io.sram.addr  := a_q.io.deq.bits.address >> log2Ceil(tlul_p.w)
  io.sram.wdata := a_q.io.deq.bits.data
  io.sram.wmask := a_q.io.deq.bits.mask

  metadata_q.io.enq.valid       := can_issue
  metadata_q.io.enq.bits.source := a_q.io.deq.bits.source
  metadata_q.io.enq.bits.size   := a_q.io.deq.bits.size
  metadata_q.io.enq.bits.opcode := a_q.io.deq.bits.opcode

  // D channel response formulation
  // Registered SRAM memory responds in the cycle after enable is asserted.
  // metadata_q stores the metadata for that response.
  val is_read = metadata_q.io.deq.bits.opcode === TLULOpcodesA.Get.asUInt

  // For registered memory, we expect rvalid to be asserted 1 cycle after enable.
  d_q.io.enq.valid       := metadata_q.io.deq.valid && io.sram.rvalid
  d_q.io.enq.bits        := 0.U.asTypeOf(d_q.io.enq.bits)
  d_q.io.enq.bits.opcode := Mux(
    is_read,
    TLULOpcodesD.AccessAckData.asUInt,
    TLULOpcodesD.AccessAck.asUInt
  )
  d_q.io.enq.bits.source := metadata_q.io.deq.bits.source
  d_q.io.enq.bits.size   := metadata_q.io.deq.bits.size
  d_q.io.enq.bits.data   := io.sram.rdata
  d_q.io.enq.bits.error  := false.B
  d_q.io.enq.bits.user   := 0.U.asTypeOf(d_q.io.enq.bits.user)

  metadata_q.io.deq.ready := d_q.io.enq.fire
}
