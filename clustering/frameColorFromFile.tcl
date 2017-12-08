#This code reads in a set of frames of the following format
# color ind1 ind2 ind3 ... indN
# and then colors the frames of the VMD file in question accordingly

proc colorFrame { fname molid } {
	mol showrep 0 0 0
	set f [open $fname r]
	gets $f lineno
	set repno 0
	for {set i 0} {$i < $lineno} {incr i} {
		gets $f line
		set c [lindex $line 0]
		set ainds [lrange $line 1 end]
		mol addrep $molid
		set repno [expr {$repno + 1}]
		mol modselect $repno $molid index $ainds and name LS
		mol modstyle $repno $molid VDW 1.0 12.0
		mol modcolor $repno $molid ColorID $c
	}
}
