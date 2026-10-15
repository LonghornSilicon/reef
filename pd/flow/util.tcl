# Common flow procs. Every stage run.tcl sources this first and builds from
# these: read, work, report, write, gate. OpenROAD commands; the Innovus
# equivalent is noted where it differs.

source $::env(REEF_CONFIG_TCL)

# Config and paths

proc cfg {key {default __none__}} {
    global cfg
    if {[info exists cfg($key)]} { return $cfg($key) }
    if {$default ne "__none__"} { return $default }
    fail "config key '$key' is not set"
}

proc log_info {msg} { puts "\[reef\] $msg" }

proc fail {msg} {
    puts stderr "\[reef\] FAIL: $msg"
    exit 1
}

proc stage_path {sub} { return [file join $::env(REEF_STAGE_DIR) $sub] }

proc prev_result {name} {
    set p [file join $::env(REEF_PREV_RESULTS) $name]
    if {![file exists $p]} { fail "previous stage did not produce $name" }
    return $p
}

# Read

# Tech and cell LEF, then one liberty per setup corner.
# Innovus: read_mmmc mmmc.tcl; read_physical -lef {...}
proc read_platform {} {
    foreach lef [cfg lef] { read_lef $lef }
    foreach lib [cfg lib.setup] { read_liberty $lib }
}

# Gate netlist, link the top, constraints. First physical stage only.
# Innovus: read_netlist $netlist -top $top; init_design
proc read_design {netlist sdc} {
    read_verilog $netlist
    link_design [cfg top]
    read_sdc $sdc
}

# Later stages start from the previous stage's database.
# Innovus: read_db
proc read_checkpoint {name} {
    read_db [prev_result $name]
}

# Report and write

# Timing and area summary into reports/<stage>_qor.txt.
# Innovus: report_timing -nworst 1; report_area
proc report_qor {} {
    set f [stage_path reports/$::env(REEF_STAGE)_qor.txt]
    report_wns > $f
    report_tns >> $f
    report_design_area >> $f
}

# Database plus DEF into results/. The next stage reads the database.
# Innovus: write_db; write_def
proc write_checkpoint {name} {
    write_db [stage_path results/$name.odb]
    write_def [stage_path results/$name.def]
}

# Gate

# gate_write {name 1 name 0 ...} writes results/gate.json. 1 is pass.
proc gate_write {checks} {
    set parts {}
    foreach {name ok} $checks {
        lappend parts "\"$name\": [expr {$ok ? "true" : "false"}]"
    }
    set f [open [stage_path results/gate.json] w]
    puts $f "{[join $parts ", "]}"
    close $f
}
