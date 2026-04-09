# tsplat

tsplat is a project for terminal based gaussian splatting rasterization, it's written in rust and rasterizes on CPU and renders direclty in the terminal. It has fast CPU and multithreaded kernels for forward rasterization making is efficient and robust

## Proof of Concept

A 120x40 terminal with half blocks is only 9600 pixels. Compare to 512x512 = 262k. CPU rasterization that would be a joke at HD becomes real time in a terminal. The bottleneck is the depth sort, not the rasterization, so let's not overengineer the kernels before profiling, let's ship an MVP first and then later we will make it more efficient

## Rendering Backends

1. Half blocks (with fg/bg truecolor), doubles vertical resolution, works everywhere. Let's ship this first
2. Quarter blocks or sextants give 2x4 subpixels per cell but color gets tricky because one cell has 8 pixels sharing one fg/bg pair          
3. Kitty/Sixel graphics, best quality but only a few terminals. Nice feature flag for later, not MVP

## Tech Stack
- glam not nalgebra, glam is meaningfully faster for graphics math
- crossterm for input and framebuffer writes, skip ratatui because you're drawing a raw framebuffer
- rayon for parallel tile rasterization
- ply-rs for loading, but be ready to patch it. INRIA .ply splats have nonstandard fields (f_dc_0..2, scale_0..2, rot_0..3, opacity). Test against the official 3DGS format and the "universal" .splat format from  
  antimatter15                                                                                                  
  - wide or nightly std::simd only after you profile and know you need it                                       
                                                                                                                
  MVP scope, nothing more                                                                                       
  1. Load .ply, convert spherical harmonics band 0 only to RGB (ignore higher bands at first, they matter less  
  than you think at terminal resolution)                                                                        
  2. Project splats to screen, sort by depth, per pixel front to back alpha composite                           
  3. Half block terminal output                                                                                 
  4. Orbit camera with hjkl or arrows, zoom with +/-                                                            
  5. FPS counter in the corner                                                                                  
                                                                                                                
  That's a weekend. Resist adding tile rasterization, SIMD, or kitty graphics until the naive version runs.     
                                                                                                                
  The non obvious gotchas                                                                                       
  - Flushing stdout per frame is the real latency killer. Build one big string per frame, single write, no per  
  cell flushes.                                                                                                 
  - 24 bit color ANSI escapes are ~20 bytes per cell. At 120x40 that's ~100KB per frame. Fine, but if you go
  bigger you'll want RLE on adjacent same color runs.                                                           
  - Trained scenes are 1-5M splats. Your CPU will cry. Ship a downsample flag (--max-splats 200000) that        
  uniformly subsamples on load. For terminal size the visual loss is basically invisible.               
  - Sort stability matters more than you think. Use sort_unstable_by with a key, depth sorting is the hot loop. 
  Radix sort is a nice later optimization but not MVP.                                                         
  - Don't bother with tile based rasterization until the per pixel version is working end to end. Tiles are an  
  optimization, not a requirement, and they add real complexity.                                              
                                                                                                                
  Things that make it actually go viral, pick one or two
  - Works over SSH. Literally the pitch: "view a 3D scene over SSH". Huge hook for HN.                          
  - cargo install splatterm && splatterm scene.ply instant demo, no build from source.                          
  - Bundled sample scene so splatterm with no args shows something beautiful. Do not make people download a     
  500MB .ply to see your project work.                                                                          
  - Asciinema recording in the README. Static screenshots undersell terminal rendering. A moving orbit is the   
  whole point.                                                                                                
