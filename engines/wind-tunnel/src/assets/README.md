# assets/

Scenes that put a **photographed subject** in the tunnel read a cut-out PNG from this folder. The
photos used in my videos aren't included, since they aren't mine to share, so bring your own:

| file | used by | what works |
|---|---|---|
| `cow.png` | `tri_shapes`, `mach_shapes` | any side-on cut-out with a transparent background, facing **right** |

`shapes.image_body()` traces the PNG's alpha channel into a solid the fluid flows around and keeps
its colours as the texture, so the subject doesn't have to be a cow. Crop tight to the subject;
tiny details (thin legs, antennae) close up at simulation resolution, which is expected.

Without the file, those two scenes stop with a message pointing here. Every other scene needs
nothing from this folder.
