// SPDX-License-Identifier: MIT OR Apache-2.0
// Print the OS's own thermal pressure, for the benchmark suite to gate on.
//
// `pmset -g therm` reports nothing on Apple Silicon and `powermetrics` needs
// root, but `ProcessInfo.thermalState` is the same signal the system uses to
// tell applications to back off, and it needs no privileges.
import Foundation

switch ProcessInfo.processInfo.thermalState {
case .nominal:  print("nominal")
case .fair:     print("fair")
case .serious:  print("serious")
case .critical: print("critical")
@unknown default: print("unknown")
}
