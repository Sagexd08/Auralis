const {
  Cam, clamp, facing, fit, prism, proj, rings, unproj, spring, stepS,
  mk, pointer, put, register, disposer, solid,
} = HL;

const COLS = 10, ROWS = 4, CELL = 14, FOOT = 11, PB = 5, UP = 9, DOWN = 2.4;
const EXT_X = COLS * CELL, EXT_Y = ROWS * CELL;
const SPACE = { c0: 2, c1: 8, r: 3 };

const falloff = (u) =>
  u <= 0 ? 1 : u <= 0.45 ? 1 - (u / 0.45) * 0.62 : u <= 1 ? 0.38 - ((u - 0.45) / 0.55) * 0.3 : 0.08;

function mount({ stage, svg, read }, value) {
  const bag = disposer();
  const C = Cam(45, 0.5, 2.05);
  fit(C, [[-6, -6, -PB], [EXT_X + 6, EXT_Y + 6, -PB], [EXT_X + 6, -6, UP], [-6, EXT_Y + 6, UP]], 200, 170);
  const P = proj(C), front = facing(C);
  let R = value * CELL, over = null;

  const g = mk("g", {}, svg);
  const [pr, pi] = rings(-6, -6, EXT_X + 6, EXT_Y + 6, 9, 2.2);
  put(solid(g), prism(P, front, pr, pi, -PB, 0));

  const keys = [];
  const add = (c0, c1, r, name) => {
    const x0 = c0 * CELL + (CELL - FOOT) / 2, x1 = c1 * CELL - (CELL - FOOT) / 2;
    const y0 = r * CELL + (CELL - FOOT) / 2, y1 = y0 + FOOT;
    const [ring, inner] = rings(x0, y0, x1, y1, 2.6, 0.9);
    const tilt = (ROWS - 1 - r) * 0.7;
    const pressed = name === "ctrl" ? UP - 4 : UP + tilt;
    keys.push({
      name, c0, c1, r, ring, inner, h0: pressed,
      cx: ((c0 + c1) / 2) * CELL, cy: (r + 0.5) * CELL,
      sp: spring(pressed, { eps: 0.03 }), el: solid(g), drawn: NaN,
    });
  };
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (r === SPACE.r && c >= SPACE.c0 && c < SPACE.c1) continue;
      add(c, c + 1, r, r === SPACE.r && c === 0 ? "ctrl" : `${r}·${c}`);
    }
  }
  add(SPACE.c0, SPACE.c1, SPACE.r, "space");
  keys.sort((a, b) => (a.c1 * CELL + (a.r + 1) * CELL) - (b.c1 * CELL + (b.r + 1) * CELL) || a.c0 - b.c0);
  keys.forEach((k) => g.appendChild(k.el.g));
  const space = keys.find((k) => k.name === "space");
  let want = space;

  const drawKey = (k) => {
    const h = Math.max(0.8, k.sp.x);
    const hi = k === want;
    if (h === k.drawn && k.hi === hi) return;
    k.drawn = h;
    k.hi = hi;
    put(k.el, prism(P, front, k.ring, k.inner, 0, h));
    k.el.sil.classList.toggle("hi", hi);
  };

  const B = register(stage, (dt) => {
    let moving = false;
    for (const k of keys) { if (stepS(k.sp, dt)) moving = true; drawKey(k); }
    return moving;
  });
  bag.add(B.unregister);

  const hit = (x, y) => {
    const c = clamp(Math.floor(x / CELL), 0, COLS - 1), r = clamp(Math.floor(y / CELL), 0, ROWS - 1);
    if (r === SPACE.r && c >= SPACE.c0 && c < SPACE.c1) return space;
    return keys.find((k) => k !== space && k.r === r && k.c0 === c);
  };

  function retarget() {
    for (const k of keys) {
      if (!over) { k.sp.t = k.h0; continue; }
      const d = Math.hypot(k.cx - over[0], k.cy - over[1]);
      k.sp.t = lerpDown(k.h0, falloff(d / R));
    }
    if (over && over[0] >= 0 && over[1] >= 0 && over[0] <= EXT_X && over[1] <= EXT_Y) {
      want = hit(over[0], over[1]);
      read.textContent = want.name === "space" ? "space" : want.name === "ctrl" ? "ctrl" : `key ${want.name}`;
    } else {
      want = space;
      read.textContent = "rest";
    }
    B.wake();
  }

  const lerpDown = (h0, share) => h0 - (h0 - DOWN) * share;

  bag.add(pointer(stage, {
    move: (p) => { over = unproj(C, p[0], p[1], 0); retarget(); },
    leave: () => { over = null; retarget(); },
  }));
  bag.add(() => svg.replaceChildren());

  return {
    set: (v) => { R = v * CELL; if (over) retarget(); },
    destroy: bag.dispose,
  };
}

hairline({
  name: "keys",
  means: "A keyboard whose keys sink under the pointer and spread out from it, the space bar lit.",
  rules: [1, 3, 5, 9],
  range: [1.2, 2.4, 4],
  mount,
});
