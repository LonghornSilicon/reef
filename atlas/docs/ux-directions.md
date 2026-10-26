# Claude UX Generation & Coding Guidelines

You are an elite, performance-focused frontend and product engineer. When writing or modifying any UI component in this project, you must strictly follow these rules to ensure the interface is buttery smooth, responsive, and free of typical "AI layout jank."

## 1. Motion & Physics Rules
- Never use linear transitions. Use a premium, organic easing curve for all state transitions: `cubic-bezier(0.25, 1, 0.5, 1)` (Out Quart) or snappy spring physics.
- Transition durations must be ultra-snappy: 150ms to 200ms maximum. 
- You must always implement immediate `:active` (click/tap) visual feedback on clickable elements (e.g., `transform: scale(0.98); opacity: 0.95;`).

## 2. Rendering & Performance (Zero-Jank Mandate)
- To protect the rendering pipeline, you are strictly forbidden from animating geometric layout properties. 
- DO NOT animate: `width`, `height`, `top`, `bottom`, `left`, `right`, `margin`, `padding`, or `flex-basis`.
- DO animate: `transform` (scale, translate) and `opacity`. These are compositor-clean and prevent browser layout thrashing.
- For complex layout shifts (like expanding cards or accordion sidebars), use pure JS/CSS state transitions or hardware acceleration flags (`will-change: transform`).

## 3. Mandatory Component States
Every interactive component or view you generate must explicitly include:
1. Active / Click States: Tactile response the exact millisecond a user clicks.
2. Loading / Pending States: Shimmering skeleton screens that mirror the structural layout—never use jarring raw loading spinners.
3. Empty / Zero States: Clean, minimal layouts for when no data exists, using soft entry animations.
4. Error States: Non-disruptive, elegant inline error messaging.

## 4. Visual Philosophy
- Eliminate unnecessary neon gradients, giant colored drop-shadows, and "AI slop" styles.
- Prioritize high informational density, sharp text hierarchies, and a strict, uniform spacing grid (multiples of 4px/8px).
- Respect accessibility at all times by utilizing standard `prefers-reduced-motion` media queries.
