[
    [ "ARM Port", "page_arm_port.html", [
      [ "ARM Port Design Document", "page_arm_port.html#autotoc_md88", [
        [ "Pattern Mode", "page_arm_port.html#autotoc_md89", [
          [ "Instrumentation to compare a memory value to an immediate", "page_arm_port.html#autotoc_md90", [
            [ "Thumb mode: can repeat single byte", "page_arm_port.html#autotoc_md91", null ],
            [ "To avoid spilling flags, try sub+cbnz in thumb mode", "page_arm_port.html#autotoc_md92", null ],
            [ "ARM mode: cannot repeat an immmed byte!  Use OP_sub x4?", "page_arm_port.html#autotoc_md93", null ],
            [ "Do 4 subtracts?", "page_arm_port.html#autotoc_md94", null ],
            [ "Switch to thumb mode just for the cmp?", "page_arm_port.html#autotoc_md95", null ],
            [ "Load immed from TLS slot", "page_arm_port.html#autotoc_md96", null ],
            [ "Go w/ unified ARM+Thumb same approach for simpler code?", "page_arm_port.html#autotoc_md97", null ],
            [ "Permanently steal another reg?", "page_arm_port.html#autotoc_md98", null ],
            [ "Put the optimizations under an option and under option switch to single-byte pattern val", "page_arm_port.html#autotoc_md99", null ],
            [ "For 2 spills, have drreg use ldm or ldrd?", "page_arm_port.html#autotoc_md100", null ]
          ] ]
        ] ]
      ] ]
    ] ]
],