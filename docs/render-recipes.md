# Recreate a video scene

Pick a scene below. Open its engine guide for setup and run the command from that engine's `src` folder. The commands render the scene visuals. For a posted edit, you may also need the original soundtrack or another source asset.

| Video or format | Render command |
| --- | --- |
| [Orbital morph](../engines/orbital-lab/) | `python cli.py render qm_orbitals --tier preview` |
| [Three plumes at different altitudes](../engines/plume-lab/) | `python cli.py render triple_altitude` |
| [Four stroke gas engine](../engines/gas-lab/) | `python cli.py single_sohc --preview` |
| [Cell colony bloom](../engines/bio-lab/) | `python cli.py render bio --scene scenes/colony_bloom.json --quality 720 --fps 30` |
| [Neon pharma profile](../engines/oscilloscope/) | `python drug_profile.py Ibuprofen --preview --no-film` |
| [Flat Chemical Scope](../engines/chemical-scope/) | `python chemical_profile.py Serotonin --preview --flat` |
| [Wind Tunnel airfoils](../engines/wind-tunnel/) | `python cli.py tri_foil_air_fast --preview` |
| [Shape Physics rings](../engines/shape-physics/) | `python cli.py escape_rings --preview` |

The engine guides link to their scene catalogues. In the engines with a scene registry, `python cli.py list` prints the available presets. Chemical Scope takes a molecule name.

Some older scenes require a cutout PNG or a song. The engine guides describe those inputs. The included Gas Lab and Orbital Lab music presets have timing data for their original tracks. Supply the matching audio yourself.
