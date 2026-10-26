(function () {
  "use strict";

  const model = window.__CORAL_MODEL__;
  if (!model) {
    document.body.innerHTML = '<main style="padding:2rem;font-family:sans-serif"><h1>Architecture model unavailable</h1><p>Run <code>python3 scripts/build.py</code> before opening the atlas.</p></main>';
    return;
  }

  const vscode = typeof acquireVsCodeApi === "function" ? acquireVsCodeApi() : null;
  if (vscode) document.body.classList.add("vscode-host");
  const priorState = vscode?.getState?.() || {};
  const byId = new Map(model.blocks.map(block => [block.id, block]));
  const owners = new Map(model.owners.map(owner => [owner.id, owner]));
  const maturityColors = { proposal: "#64748b", specified: "#087f8c", "in-progress": "#b45309", implemented: "#2563eb", verified: "#2f855a" };
  const kindColors = { system: "#101c26", chip: "#101c26", compute: "#ef3434", memory: "#d97706", engine: "#55a63f", interconnect: "#6d28d9", controller: "#087f8c", bridge: "#9aabb5", arbiter: "#55a63f", peripheral: "#475569", "peripheral-group": "#475569", control: "#2563eb", datapath: "#ef3434", interface: "#6d28d9", "external-memory": "#c25700" };
  const icons = {
    search: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="6.5"/><path d="m16 16 4 4"/></svg>',
    close: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18"/></svg>',
    blocks: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="7" height="6" rx="1"/><rect x="14" y="4" width="7" height="6" rx="1"/><rect x="8.5" y="15" width="7" height="5" rx="1"/><path d="M6.5 10v2.5h11V10M12 12.5V15"/></svg>',
    flow: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="5" cy="6" r="2"/><circle cx="19" cy="18" r="2"/><path d="M7 6h5a3 3 0 0 1 3 3v6M11 12l4 3 4-3"/></svg>',
    protocols: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 6h10M4 12h16M10 18h10"/><circle cx="17" cy="6" r="3"/><circle cx="7" cy="18" r="3"/></svg>',
    instructions: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="6" height="5" rx="1"/><rect x="15" y="4" width="6" height="5" rx="1"/><rect x="9" y="15" width="6" height="5" rx="1"/><path d="M9 6.5h6M18 9v3h-6v3"/></svg>',
    people: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="9" cy="8" r="3"/><circle cx="17" cy="9" r="2.5"/><path d="M3.5 19c.5-4 2.4-6 5.5-6s5 2 5.5 6M14 14c3.7-.7 5.8 1 6.5 5"/></svg>',
    download: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v12m-4-4 4 4 4-4M5 19h14"/></svg>',
    image: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="8" cy="9" r="2"/><path d="m5 18 5-5 3 3 2-2 4 4"/></svg>',
    book: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 5.5A3.5 3.5 0 0 1 7.5 2H11v17H7.5A3.5 3.5 0 0 0 4 22V5.5ZM20 5.5A3.5 3.5 0 0 0 16.5 2H13v17h3.5A3.5 3.5 0 0 1 20 22V5.5Z"/></svg>',
    fit: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5"/></svg>',
    enter: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 5 7 7-7 7"/></svg>',
    back: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m15 5-7 7 7 7"/></svg>',
    up: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 15 7-7 7 7"/></svg>',
    down: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 9 7 7 7-7"/></svg>',
    play: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m8 5 11 7-11 7z"/></svg>',
    pause: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14M16 5v14"/></svg>',
    move: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v18M3 12h18M8 7l4-4 4 4M8 17l4 4 4-4M7 8l-4 4 4 4M17 8l4 4-4 4"/></svg>',
    copy: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="11" height="11" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/></svg>',
    reset: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7v5h5M5.5 17a8 8 0 1 0-.8-9.2L4 12"/></svg>',
    previous: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5v14M18 6l-8 6 8 6z"/></svg>',
    next: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M17 5v14M6 6l8 6-8 6z"/></svg>',
    file: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 3h8l4 4v14H6zM14 3v5h4M9 12h6M9 16h6"/></svg>',
    external: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 5h5v5M19 5l-8 8M18 13v6H5V6h6"/></svg>'
  };

  document.querySelectorAll("[data-icon]").forEach(element => {
    const key = element.dataset.icon;
    if (icons[key]) element.insertAdjacentHTML("afterbegin", icons[key]);
  });

  const els = Object.fromEntries(["diagram-viewport", "diagram-stage", "edge-layer", "node-layer", "breadcrumbs", "model-count", "protocol-legend", "inspector-empty", "inspector-content", "empty-state", "zoom-value", "search-layer", "search-input", "search-results", "status-filter", "owner-filter", "toast", "sync-indicator", "arrange-toggle", "arrange-actions", "hierarchy-nav", "nav-up", "nav-down"].map(id => [id.replaceAll("-", "_"), document.getElementById(id)]));
  const urlState = readHash();
  const layoutEdits = loadLayoutEdits();
  const state = {
    view: urlState.view || priorState.view || "structure",
    focus: byId.has(urlState.focus) ? urlState.focus : (byId.has(priorState.focus) ? priorState.focus : model.product.default_focus),
    selected: byId.has(urlState.selected) ? urlState.selected : (byId.has(priorState.selected) ? priorState.selected : ""),
    zoom: 1,
    query: "",
    status: "",
    owner: "",
    arrange: false,
    instructionFlow: model.instruction_flows?.some(flow => flow.id === urlState.instruction) ? urlState.instruction : (model.instruction_flows?.some(flow => flow.id === priorState.instructionFlow) ? priorState.instructionFlow : (model.instruction_flows?.[0]?.id || ""))
  };
  let toastTimer;
  let firstStructureRender = true;
  let searchOpener = null;
  let bootStep = 0;
  let bootTimer = null;
  let bootPlaying = false;
  let activeBootBehavior = null;
  let activeStructure = null;
  let activeGesture = null;
  let spacePressed = false;
  let suppressNextClick = false;

  function loadLayoutEdits() {
    if (priorState.layoutSchema === 2 && priorState.layoutEdits && typeof priorState.layoutEdits === "object") return priorState.layoutEdits;
    try {
      const stored = JSON.parse(localStorage.getItem("coral-layout-edits-v2") || "{}");
      return stored && typeof stored === "object" ? stored : {};
    } catch {
      return {};
    }
  }

  function readHash() {
    const params = new URLSearchParams(location.hash.replace(/^#/, ""));
    return Object.fromEntries(params.entries());
  }

  function persist() {
    const params = new URLSearchParams({ view: state.view });
    if (state.focus) params.set("focus", state.focus);
    if (state.selected) params.set("selected", state.selected);
    if (state.instructionFlow) params.set("instruction", state.instructionFlow);
    history.replaceState(null, "", `#${params}`);
    vscode?.setState?.({ view: state.view, focus: state.focus, selected: state.selected, instructionFlow: state.instructionFlow, layoutSchema: 2, layoutEdits });
    try { localStorage.setItem("coral-layout-edits-v2", JSON.stringify(layoutEdits)); } catch { /* Storage can be disabled in embedded hosts. */ }
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
  }

  function childrenOf(id) { return model.blocks.filter(block => block.parent === id).sort((a, b) => a.rank - b.rank || a.order - b.order || a.name.localeCompare(b.name)); }
  function descendantsOf(id) { const result = []; const visit = parent => childrenOf(parent).forEach(child => { result.push(child); visit(child.id); }); visit(id); return result; }
  function ancestorsOf(id) { const chain = []; let item = byId.get(id); while (item) { chain.unshift(item); item = item.parent ? byId.get(item.parent) : null; } return chain; }
  function hasChildren(id) { return model.blocks.some(block => block.parent === id); }
  function stateLabel(value) { return value.replaceAll("-", " "); }

  function setView(view) {
    if (!["structure", "boot", "protocols", "instructions", "ownership"].includes(view)) return;
    stopBootPlayback();
    state.view = view;
    if (view !== "structure") state.arrange = false;
    document.querySelectorAll("[data-view]").forEach(button => {
      const active = button.dataset.view === view;
      button.classList.toggle("active", active);
      if (button.matches("[role=tab]")) button.setAttribute("aria-selected", String(active));
    });
    persist();
    render();
  }

  function setFocus(id) {
    if (!byId.has(id)) return;
    state.focus = id;
    state.selected = "";
    state.zoom = 1;
    persist();
    render();
    requestAnimationFrame(() => fitDiagram());
  }

  function focusParent() {
    const parent = byId.get(state.focus)?.parent;
    if (parent) setFocus(parent);
  }

  function enterBlock(id) {
    if (!hasChildren(id)) return;
    if (state.view !== "structure") state.view = "structure";
    stopBootPlayback();
    setFocus(id);
  }

  function selectBlock(id, options = {}) {
    const block = byId.get(id);
    if (!block) return;
    state.selected = id;
    persist();
    renderInspector(block);
    document.querySelectorAll(".diagram-node").forEach(node => node.classList.toggle("selected", node.dataset.id === id));
    document.querySelectorAll(".internal-node").forEach(node => node.classList.toggle("selected", node.dataset.child === id));
    updateHierarchyNavigation();
    if (options.openSource && vscode) vscode.postMessage({ type: "openSymbol", id });
    if (window.innerWidth <= 820) requestAnimationFrame(() => document.getElementById("inspector").scrollIntoView({ block: "start", behavior: "smooth" }));
  }

  function layoutBlocks(blocks, focusId) {
    const authored = model.layouts[focusId];
    if (!authored) return { positions: new Map(), width: 960, height: 620 };
    const positions = new Map();
    blocks.forEach(block => {
      const position = authored.nodes[block.id];
      const edited = layoutEdits[focusId]?.[block.id];
      if (position) positions.set(block.id, {
        x: Number.isFinite(edited?.x) ? edited.x : position.x,
        y: Number.isFinite(edited?.y) ? edited.y : position.y,
        w: position.width,
        h: position.height
      });
    });
    return { positions, width: authored.canvas.width, height: authored.canvas.height };
  }

  function portFor(blockId, portId) {
    return model.ports[blockId]?.[portId];
  }

  function portPoint(position, port) {
    const offset = port.offset;
    if (port.side === "left") return { x: position.x, y: position.y + position.h * offset };
    if (port.side === "right") return { x: position.x + position.w, y: position.y + position.h * offset };
    if (port.side === "top") return { x: position.x + position.w * offset, y: position.y };
    return { x: position.x + position.w * offset, y: position.y + position.h };
  }

  function offsetPoint(point, side, distance) {
    if (side === "left") return { x: point.x - distance, y: point.y };
    if (side === "right") return { x: point.x + distance, y: point.y };
    if (side === "top") return { x: point.x, y: point.y - distance };
    return { x: point.x, y: point.y + distance };
  }

  function segmentBlocked(a, b, obstacles) {
    const epsilon = .01;
    if (Math.abs(a.y - b.y) < epsilon) {
      const left = Math.min(a.x, b.x), right = Math.max(a.x, b.x);
      return obstacles.some(rect => a.y > rect.top + epsilon && a.y < rect.bottom - epsilon && right > rect.left + epsilon && left < rect.right - epsilon);
    }
    const top = Math.min(a.y, b.y), bottom = Math.max(a.y, b.y);
    return obstacles.some(rect => a.x > rect.left + epsilon && a.x < rect.right - epsilon && bottom > rect.top + epsilon && top < rect.bottom - epsilon);
  }

  function compressRoute(points) {
    const compact = [];
    points.forEach(point => {
      const previous = compact.at(-1);
      if (!previous || previous.x !== point.x || previous.y !== point.y) compact.push(point);
    });
    return compact.filter((point, index) => {
      if (index === 0 || index === compact.length - 1) return true;
      const before = compact[index - 1], after = compact[index + 1];
      return !((before.x === point.x && point.x === after.x) || (before.y === point.y && point.y === after.y));
    });
  }

  function routeFor(start, end, fromSide, toSide, options = {}) {
    const clearance = 12;
    const startExit = offsetPoint(start, fromSide, clearance + 2);
    const endExit = offsetPoint(end, toSide, clearance + 2);
    const width = options.width || Math.max(start.x, end.x) + 100;
    const height = options.height || Math.max(start.y, end.y) + 100;
    const obstacles = (options.obstacles || []).map(position => ({
      left: position.x - clearance,
      right: position.x + position.w + clearance,
      top: position.y - clearance,
      bottom: position.y + position.h + clearance
    }));
    const xs = new Set([startExit.x, endExit.x, 4, width - 4]);
    const ys = new Set([startExit.y, endExit.y, 4, height - 4]);
    obstacles.forEach(rect => {
      xs.add(Math.max(4, rect.left));
      xs.add(Math.min(width - 4, rect.right));
      ys.add(Math.max(4, rect.top));
      ys.add(Math.min(height - 4, rect.bottom));
    });
    const xValues = [...xs].filter(value => value >= 0 && value <= width).sort((a, b) => a - b);
    const yValues = [...ys].filter(value => value >= 0 && value <= height).sort((a, b) => a - b);
    const points = [];
    const pointIndex = new Map();
    const pointIsBlocked = point => obstacles.some(rect => point.x > rect.left && point.x < rect.right && point.y > rect.top && point.y < rect.bottom);
    yValues.forEach(y => xValues.forEach(x => {
      const point = { x, y };
      if (pointIsBlocked(point)) return;
      pointIndex.set(`${x}|${y}`, points.length);
      points.push(point);
    }));
    const startIndex = pointIndex.get(`${startExit.x}|${startExit.y}`);
    const endIndex = pointIndex.get(`${endExit.x}|${endExit.y}`);
    if (startIndex === undefined || endIndex === undefined) return { d: `M ${start.x} ${start.y} L ${startExit.x} ${startExit.y} L ${endExit.x} ${endExit.y} L ${end.x} ${end.y}` };

    const neighbors = Array.from({ length: points.length }, () => []);
    const connectLine = indexes => {
      indexes.sort((left, right) => points[left].x - points[right].x || points[left].y - points[right].y);
      for (let index = 1; index < indexes.length; index += 1) {
        const aIndex = indexes[index - 1], bIndex = indexes[index];
        const a = points[aIndex], b = points[bIndex];
        if (segmentBlocked(a, b, obstacles)) continue;
        const distance = Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
        const direction = a.x === b.x ? "V" : "H";
        neighbors[aIndex].push({ index: bIndex, distance, direction });
        neighbors[bIndex].push({ index: aIndex, distance, direction });
      }
    };
    yValues.forEach(y => connectLine(points.map((point, index) => point.y === y ? index : -1).filter(index => index >= 0)));
    xValues.forEach(x => connectLine(points.map((point, index) => point.x === x ? index : -1).filter(index => index >= 0)));

    const startKey = `${startIndex}|N`;
    const distances = new Map([[startKey, 0]]);
    const previous = new Map();
    const queue = [{ key: startKey, index: startIndex, direction: "N", cost: 0 }];
    let finalKey = null;
    while (queue.length) {
      queue.sort((left, right) => left.cost - right.cost);
      const current = queue.shift();
      if (current.cost !== distances.get(current.key)) continue;
      if (current.index === endIndex) { finalKey = current.key; break; }
      neighbors[current.index].forEach(next => {
        const turnCost = current.direction !== "N" && current.direction !== next.direction ? 18 : 0;
        const cost = current.cost + next.distance + turnCost;
        const key = `${next.index}|${next.direction}`;
        if (cost >= (distances.get(key) ?? Infinity)) return;
        distances.set(key, cost);
        previous.set(key, current.key);
        queue.push({ key, index: next.index, direction: next.direction, cost });
      });
    }
    if (!finalKey) return { d: `M ${start.x} ${start.y} L ${startExit.x} ${startExit.y} L ${endExit.x} ${endExit.y} L ${end.x} ${end.y}` };
    const routed = [];
    for (let key = finalKey; key; key = previous.get(key)) routed.unshift(points[Number(key.split("|")[0])]);
    const routePoints = compressRoute([start, startExit, ...routed, endExit, end]);
    return { d: routePoints.map((point, index) => `${index ? "L" : "M"} ${point.x} ${point.y}`).join(" ") };
  }

  function portBadges(blockId, compact = false) {
    return Object.entries(model.ports[blockId] || {}).map(([id, port]) => {
      const protocol = model.protocols[port.protocol] || model.protocols.control;
      return `<span class="port-badge side-${port.side}${compact ? " compact" : ""}" data-port="${escapeHtml(id)}" style="--port-offset:${port.offset * 100}%;--port-color:${protocol.color}">${escapeHtml(port.label)}</span>`;
    }).join("");
  }

  function portSideClasses(blockId) {
    return [...new Set(Object.values(model.ports[blockId] || {}).map(port => port.side))].map(side => ` ports-${side}`).join("");
  }

  function proportionalTitleSize(position, compact = false) {
    const widthScale = position.width / (compact ? 12 : 16);
    const heightScale = position.height / (compact ? 6 : 8);
    return Math.max(compact ? 2 : 10, Math.min(compact ? 8 : 22, widthScale, heightScale));
  }

  function internalNodeMarkup(parentId, child, parentPosition) {
    const authored = model.layouts[parentId];
    const position = authored.nodes[child.id];
    const canvas = authored.canvas;
    const parentPorts = new Set(Object.values(model.ports[parentId] || {}).map(port => port.side));
    const gridWidth = Math.max(1, parentPosition.w - 12 - (parentPorts.has("left") ? 72 : 0) - (parentPorts.has("right") ? 72 : 0));
    const gridHeight = Math.max(1, parentPosition.h - 65 - (parentPorts.has("bottom") ? 40 : 0));
    const renderedPosition = { width: position.width / canvas.width * gridWidth, height: position.height / canvas.height * gridHeight };
    const titleSize = proportionalTitleSize(renderedPosition, true);
    const style = [
      `--node-accent:${kindColors[child.kind] || "#64748b"}`,
      `left:${position.x / canvas.width * 100}%`,
      `top:${position.y / canvas.height * 100}%`,
      `width:${position.width / canvas.width * 100}%`,
      `height:${position.height / canvas.height * 100}%`,
      `--internal-title-size:${titleSize.toFixed(2)}px`,
      `--internal-meta-size:${Math.max(2.2, Math.min(5, titleSize * .58)).toFixed(2)}px`,
      `--internal-port-size:${Math.max(2.2, Math.min(4, titleSize * .48)).toFixed(2)}px`
    ].join(";");
    return `<button class="internal-node${hasChildren(child.id) ? " outlined" : ""}${state.selected === child.id ? " selected" : ""}${portSideClasses(child.id)}" type="button" data-child="${escapeHtml(child.id)}" style="${style}" title="${escapeHtml(child.name)}"><span class="internal-name">${escapeHtml(child.name)}</span>${hasChildren(child.id) ? `<span class="internal-enter-icon" aria-hidden="true">${icons.enter}</span>` : ""}${portBadges(child.id, true)}</button>`;
  }

  function internalConnectionsMarkup(parentId) {
    const layout = model.layouts[parentId];
    if (!layout) return "";
    const childIds = new Set(childrenOf(parentId).map(child => child.id));
    const obstacles = [...childIds].map(id => {
      const position = layout.nodes[id];
      return position ? { x: position.x, y: position.y, w: position.width, h: position.height } : null;
    }).filter(Boolean);
    const connections = model.connections.filter(connection => childIds.has(connection.from) && childIds.has(connection.to));
    const paths = connections.map(connection => {
      const fromRaw = layout.nodes[connection.from];
      const toRaw = layout.nodes[connection.to];
      const fromPort = portFor(connection.from, connection.from_port);
      const toPort = portFor(connection.to, connection.to_port);
      if (!fromRaw || !toRaw || !fromPort || !toPort) return "";
      const from = { x: fromRaw.x, y: fromRaw.y, w: fromRaw.width, h: fromRaw.height };
      const to = { x: toRaw.x, y: toRaw.y, w: toRaw.width, h: toRaw.height };
      const route = routeFor(portPoint(from, fromPort), portPoint(to, toPort), fromPort.side, toPort.side, { obstacles, width: layout.canvas.width, height: layout.canvas.height });
      const protocol = model.protocols[connection.protocol] || model.protocols.control;
      const attributes = `data-connection="${escapeHtml(connection.id)}" data-from="${escapeHtml(connection.from)}" data-to="${escapeHtml(connection.to)}"`;
      return `<path class="internal-edge" ${attributes} d="${route.d}" stroke="${protocol.color}"/>${connection.bidirectional ? `<path class="internal-edge return" ${attributes} d="${route.d}" stroke="${protocol.color}"/>` : ""}`;
    }).join("");
    return `<svg class="internal-edge-layer" viewBox="0 0 ${layout.canvas.width} ${layout.canvas.height}" preserveAspectRatio="none" aria-hidden="true">${paths}</svg>`;
  }

  function visibleConnections(blocks) {
    const ids = new Set(blocks.map(block => block.id));
    return model.connections.filter(connection => ids.has(connection.from) && ids.has(connection.to));
  }

  function drawOuterConnections(blocks, layout) {
    els.edge_layer.replaceChildren();
    const obstacles = [...layout.positions.values()];
    visibleConnections(blocks).forEach(connection => {
      const from = layout.positions.get(connection.from);
      const to = layout.positions.get(connection.to);
      if (!from || !to) return;
      const fromPort = portFor(connection.from, connection.from_port);
      const toPort = portFor(connection.to, connection.to_port);
      if (!fromPort || !toPort) return;
      const route = routeFor(portPoint(from, fromPort), portPoint(to, toPort), fromPort.side, toPort.side, { obstacles, width: layout.width, height: layout.height });
      const protocol = model.protocols[connection.protocol] || model.protocols.control;
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", route.d);
      path.setAttribute("class", "edge");
      path.setAttribute("stroke", protocol.color);
      path.dataset.connection = connection.id;
      path.dataset.from = connection.from;
      path.dataset.to = connection.to;
      els.edge_layer.appendChild(path);
      if (connection.bidirectional) {
        const returnPath = path.cloneNode();
        returnPath.setAttribute("class", "edge return");
        els.edge_layer.appendChild(returnPath);
      }
    });
  }

  function updateBoundaryFrame(focus, blocks, layout) {
    const boundary = els.node_layer.querySelector(".boundary-frame");
    if (!boundary) return;
    const contained = blocks.filter(block => block.kind !== "external-memory").map(block => layout.positions.get(block.id)).filter(Boolean);
    if (contained.length < 2) { boundary.hidden = true; return; }
    boundary.hidden = false;
    const enclosureGutter = 32;
    const left = Math.min(...contained.map(item => item.x)) - enclosureGutter;
    const top = Math.min(...contained.map(item => item.y)) - enclosureGutter;
    const right = Math.max(...contained.map(item => item.x + item.w)) + enclosureGutter;
    const bottom = Math.max(...contained.map(item => item.y + item.h)) + enclosureGutter;
    Object.assign(boundary.style, { left: `${left}px`, top: `${top}px`, width: `${right - left}px`, height: `${bottom - top}px` });
    boundary.querySelector("span").textContent = focus.name;
  }

  function refreshStructureGeometry() {
    if (!activeStructure) return;
    drawOuterConnections(activeStructure.blocks, activeStructure.layout);
    updateBoundaryFrame(activeStructure.focus, activeStructure.blocks, activeStructure.layout);
  }

  function renderStructure(options = {}) {
    const bootBehavior = options.bootBehavior || null;
    if (!bootBehavior) {
      stopBootPlayback();
      activeBootBehavior = null;
    }
    document.getElementById("boot-player")?.remove();
    document.querySelector(".canvas-panel")?.classList.toggle("boot-mode", Boolean(bootBehavior));
    els.diagram_viewport.className = `diagram-viewport can-pan${bootBehavior ? " boot-diagram-mode" : ""}`;
    els.diagram_viewport.innerHTML = '<div class="diagram-stage-shell" id="diagram-stage-shell"><div class="diagram-stage" id="diagram-stage"><svg class="edge-layer" id="edge-layer" aria-hidden="true"></svg><div class="node-layer" id="node-layer"></div></div></div><div class="empty-state" id="empty-state" hidden><div class="empty-symbol" aria-hidden="true"></div><h2>No matching blocks</h2><p>Clear the filters or move back up one level.</p></div>';
    els.diagram_stage_shell = document.getElementById("diagram-stage-shell");
    els.diagram_stage = document.getElementById("diagram-stage");
    els.edge_layer = document.getElementById("edge-layer");
    els.node_layer = document.getElementById("node-layer");
    els.empty_state = document.getElementById("empty-state");
    const focus = byId.get(options.focusId || state.focus) || byId.get(model.product.default_focus);
    const children = childrenOf(focus.id);
    const childIds = new Set(children.map(block => block.id));
    const contextBlocks = focus.id === "coral-soc" ? model.connections.flatMap(connection => {
      const from = byId.get(connection.from);
      const to = byId.get(connection.to);
      if (childIds.has(connection.from) && to?.kind === "external-memory") return [{ ...to, rank: -1 }];
      if (childIds.has(connection.to) && from?.kind === "external-memory") return [{ ...from, rank: -1 }];
      return [];
    }) : [];
    const blocks = children.length ? [...new Map([...contextBlocks, ...children].map(block => [block.id, block])).values()] : [focus];
    const layout = layoutBlocks(blocks, focus.id);
    els.diagram_stage.style.width = `${layout.width}px`;
    els.diagram_stage.style.height = `${layout.height}px`;
    els.diagram_stage.style.setProperty("--zoom", state.zoom);
    els.edge_layer.setAttribute("viewBox", `0 0 ${layout.width} ${layout.height}`);
    drawOuterConnections(blocks, layout);

    const boundary = document.createElement("div");
    boundary.className = "boundary-frame";
    boundary.innerHTML = `<span>${escapeHtml(focus.name)}</span>`;
    els.node_layer.appendChild(boundary);
    updateBoundaryFrame(focus, blocks, layout);

    blocks.forEach(block => {
      const pos = layout.positions.get(block.id);
      if (!pos) return;
      const internal = childrenOf(block.id);
      const node = document.createElement(internal.length ? "section" : "button");
      if (!internal.length) node.type = "button";
      node.className = `diagram-node kind-${block.kind}${internal.length ? " composite" : " leaf"}${state.selected === block.id ? " selected" : ""}${portSideClasses(block.id)}`;
      node.dataset.id = block.id;
      node.tabIndex = 0;
      node.style.left = `${pos.x}px`;
      node.style.top = `${pos.y}px`;
      node.style.width = `${pos.w}px`;
      node.style.height = `${pos.h}px`;
      node.style.setProperty("--node-accent", kindColors[block.kind] || "#64748b");
      const proportionalSize = proportionalTitleSize({ width: pos.w, height: pos.h });
      const titleSize = internal.length ? Math.min(18, proportionalSize) : proportionalSize;
      node.style.setProperty("--node-title-size", `${titleSize.toFixed(2)}px`);
      node.style.setProperty("--node-kind-size", `${Math.max(6, Math.min(9, titleSize * .52)).toFixed(2)}px`);
      node.style.setProperty("--node-port-size", `${Math.max(5.5, Math.min(8, titleSize * .44)).toFixed(2)}px`);
      node.setAttribute("aria-label", `${block.name}, ${block.kind}, ${stateLabel(block.maturity)}${internal.length ? `, contains ${internal.length} blocks` : ""}${state.arrange ? ", movable with arrow keys" : ""}`);
      if (internal.length) {
        node.innerHTML = `<div class="composite-header"><button class="composite-select" type="button"><span><strong>${escapeHtml(block.name)}</strong><small>${escapeHtml(block.kind.replaceAll("-", " "))} · ${internal.length} blocks</small></span></button><button class="composite-enter" type="button" aria-label="Open ${escapeHtml(block.name)} internals" title="Open ${escapeHtml(block.name)} internals">${icons.enter}</button></div><div class="internal-grid${internal.length > 8 ? " dense" : ""}">${internalConnectionsMarkup(block.id)}${internal.map(child => internalNodeMarkup(block.id, child, pos)).join("")}</div>${portBadges(block.id)}`;
        node.querySelector(".composite-select").addEventListener("click", () => selectBlock(block.id, { openSource: true }));
        node.querySelector(".composite-enter").addEventListener("click", () => enterBlock(block.id));
        node.querySelectorAll("[data-child]").forEach(child => child.addEventListener("click", event => { event.stopPropagation(); selectBlock(child.dataset.child, { openSource: true }); }));
      } else {
        node.innerHTML = `<span class="node-body"><span class="node-name">${escapeHtml(block.name)}</span><span class="node-kind">${escapeHtml(block.kind.replaceAll("-", " "))}</span></span>${portBadges(block.id)}`;
        node.addEventListener("click", () => selectBlock(block.id, { openSource: true }));
      }
      els.node_layer.appendChild(node);
    });
    activeStructure = { focus, blocks, layout };
    updateArrangeUi();
    renderBreadcrumbs(focus.id);
    const nestedCount = blocks.reduce((sum, block) => sum + childrenOf(block.id).length, 0);
    els.model_count.textContent = `${blocks.length} blocks${nestedCount ? ` · ${nestedCount} nested` : ""} · ${visibleConnections(blocks).length} links`;
    els.empty_state.hidden = blocks.length > 0;
    renderLegend();
    if (bootBehavior) installBootPlayer(bootBehavior);
    if (firstStructureRender) {
      firstStructureRender = false;
      requestAnimationFrame(() => fitDiagram());
    }
  }

  function renderBreadcrumbs(id) {
    els.breadcrumbs.innerHTML = ancestorsOf(id).map((item, index, all) => `<button class="crumb${index === all.length - 1 ? " current" : ""}" type="button" data-focus="${escapeHtml(item.id)}"${index === all.length - 1 ? ' aria-current="page"' : ""}>${escapeHtml(item.name)}</button>${index < all.length - 1 ? '<span class="crumb-separator">/</span>' : ""}`).join("");
    els.breadcrumbs.querySelectorAll("[data-focus]").forEach(button => button.addEventListener("click", () => setFocus(button.dataset.focus)));
  }

  function updateHierarchyNavigation() {
    const available = state.view === "structure";
    els.hierarchy_nav.hidden = !available;
    if (!available) return;
    const parent = byId.get(state.focus)?.parent;
    const selected = byId.get(state.selected);
    const downTarget = selected && hasChildren(selected.id) ? selected : null;
    els.nav_up.disabled = !parent;
    els.nav_up.title = parent ? `Up to ${byId.get(parent)?.name || parent}` : "Already at the top level";
    els.nav_down.disabled = !downTarget;
    els.nav_down.title = downTarget ? `Open ${downTarget.name}` : "Select a block with internals to move down";
  }

  function renderLegend() {
    els.protocol_legend.hidden = false;
    els.protocol_legend.innerHTML = Object.entries(model.protocols).filter(([id]) => id !== "containment").map(([, protocol]) => `<span class="legend-item"><span class="legend-line" style="--legend:${protocol.color}"></span>${escapeHtml(protocol.abbreviation)}</span>`).join("");
  }

  function resetReferenceView(className) {
    stopBootPlayback();
    activeBootBehavior = null;
    activeStructure = null;
    document.getElementById("boot-player")?.remove();
    document.querySelector(".canvas-panel")?.classList.remove("boot-mode");
    els.diagram_viewport.className = `diagram-viewport reference-stage ${className}`;
    els.diagram_viewport.scrollTo({ left: 0, top: 0, behavior: "auto" });
  }

  function renderProtocols() {
    resetReferenceView("protocol-stage");
    const protocols = Object.entries(model.protocols).filter(([id]) => id !== "containment");
    els.diagram_viewport.innerHTML = `<div class="reference-page"><header class="reference-heading"><h1>Protocols and signal paths</h1><p>Colors in the schematic identify transport semantics. Endpoint badges name the role each block plays on that interface.</p></header><div class="protocol-catalog">${protocols.map(([id, protocol]) => {
      const connections = model.connections.filter(connection => connection.protocol === id);
      const roles = protocol.roles || [];
      return `<article class="protocol-row" style="--protocol:${protocol.color}"><div class="protocol-identity"><span class="protocol-swatch" aria-hidden="true"></span><strong>${escapeHtml(protocol.abbreviation)}</strong><span>${escapeHtml(protocol.name)}</span></div><p>${escapeHtml(protocol.purpose || "Architecture interface.")}</p><div class="protocol-roles" aria-label="Endpoint roles">${roles.map(role => `<span>${escapeHtml(role)}</span>`).join('<span class="role-arrow" aria-hidden="true">→</span>')}</div><details><summary>${connections.length} mapped link${connections.length === 1 ? "" : "s"}</summary><ul>${connections.map(connection => `<li><button type="button" data-block="${escapeHtml(connection.from)}">${escapeHtml(byId.get(connection.from)?.name || connection.from)}</button><span>→</span><button type="button" data-block="${escapeHtml(connection.to)}">${escapeHtml(byId.get(connection.to)?.name || connection.to)}</button><small>${escapeHtml(connection.label)}</small></li>`).join("") || "<li>No instantiated links yet.</li>"}</ul></details></article>`;
    }).join("")}</div></div>`;
    els.diagram_viewport.querySelectorAll("[data-block]").forEach(button => button.addEventListener("click", () => selectBlock(button.dataset.block, { openSource: true })));
    els.breadcrumbs.innerHTML = '<span class="crumb current" aria-current="page">Protocols</span>';
    els.model_count.textContent = `${protocols.length} protocols · ${model.connections.length} mapped links`;
    els.protocol_legend.innerHTML = "";
    els.protocol_legend.hidden = true;
  }

  function renderInstructions() {
    resetReferenceView("instruction-stage");
    const flows = model.instruction_flows || [];
    const flow = flows.find(candidate => candidate.id === state.instructionFlow) || flows[0];
    if (!flow) {
      els.diagram_viewport.innerHTML = '<div class="empty-state"><h2>No instruction paths</h2><p>Add instruction_flows to architecture/soc.yaml.</p></div>';
      return;
    }
    state.instructionFlow = flow.id;
    const usedProtocols = new Set(flow.steps.map(step => step.protocol));
    els.diagram_viewport.innerHTML = `<div class="reference-page instruction-page"><header class="reference-heading"><h1>Instruction execution</h1><p>These are architecture-level planned paths derived from the current Core, Scalar, and Vector reference diagrams—not cycle-accurate RTL timing.</p></header><div class="instruction-layout"><nav class="instruction-picker" aria-label="Instruction families">${flows.map(candidate => `<button type="button" data-instruction-flow="${escapeHtml(candidate.id)}"${candidate.id === flow.id ? ' aria-current="true"' : ""}><strong>${escapeHtml(candidate.name)}</strong><span>${escapeHtml(candidate.summary)}</span></button>`).join("")}</nav><section class="execution-board" aria-labelledby="execution-title"><header><h2 id="execution-title">${escapeHtml(flow.name)}</h2><p>${escapeHtml(flow.summary)}</p></header><ol class="execution-rail">${flow.steps.map((step, index) => {
      const protocol = model.protocols[step.protocol] || model.protocols.control;
      return `<li style="--step-color:${protocol.color}"><span class="execution-index">${index + 1}</span><div class="execution-copy"><div><h3>${escapeHtml(step.title)}</h3><span class="execution-protocol">${escapeHtml(protocol.abbreviation)}</span></div><p>${escapeHtml(step.detail)}</p><div class="execution-blocks">${step.blocks.map(id => `<button type="button" data-block="${escapeHtml(id)}">${escapeHtml(byId.get(id)?.name || id)}</button>`).join("")}</div></div></li>`;
    }).join("")}</ol></section></div></div>`;
    els.diagram_viewport.querySelectorAll("[data-instruction-flow]").forEach(button => button.addEventListener("click", () => {
      state.instructionFlow = button.dataset.instructionFlow;
      persist();
      renderInstructions();
      updateHierarchyNavigation();
    }));
    els.diagram_viewport.querySelectorAll("[data-block]").forEach(button => button.addEventListener("click", () => selectBlock(button.dataset.block, { openSource: true })));
    els.breadcrumbs.innerHTML = `<span class="crumb">Instruction execution</span><span class="crumb-separator">/</span><span class="crumb current" aria-current="page">${escapeHtml(flow.name)}</span>`;
    els.model_count.textContent = `${flow.steps.length} stages · ${flow.steps.reduce((sum, step) => sum + step.blocks.length, 0)} block references`;
    els.protocol_legend.innerHTML = [...usedProtocols].map(id => {
      const protocol = model.protocols[id];
      return protocol ? `<span class="legend-item"><span class="legend-line" style="--legend:${protocol.color}"></span>${escapeHtml(protocol.abbreviation)}</span>` : "";
    }).join("");
    els.protocol_legend.hidden = false;
  }

  function installBootPlayer(behavior) {
    stopBootPlayback();
    activeBootBehavior = behavior;
    bootStep = Math.min(bootStep, behavior.steps.length - 1);
    const player = document.createElement("section");
    player.id = "boot-player";
    player.className = "boot-player";
    player.setAttribute("aria-label", "Boot flow playback");
    player.innerHTML = `<div class="boot-player-copy"><div class="boot-step-meta"><strong id="boot-step-count"></strong><span id="boot-step-mode"></span></div><h2 id="boot-step-title"></h2><p id="boot-step-detail"></p><div class="boot-step-blocks" id="boot-step-blocks"></div></div><div class="boot-player-controls"><button type="button" data-boot="previous" aria-label="Previous boot step">${icons.previous}</button><button class="boot-toggle" type="button" data-boot="toggle" aria-label="Play boot animation">${icons.play}<span>Play</span></button><button type="button" data-boot="next" aria-label="Next boot step">${icons.next}</button></div><div class="boot-progress" aria-label="Boot sequence steps">${behavior.steps.map((step, index) => `<button type="button" data-boot-step="${index}" aria-label="Go to step ${index + 1}: ${escapeHtml(step.title)}"><span></span></button>`).join("")}</div><div class="sr-only" id="boot-announcement" aria-live="polite"></div>`;
    document.querySelector(".canvas-panel").appendChild(player);
    player.querySelector('[data-boot="previous"]').addEventListener("click", () => { stopBootPlayback(); setBootStep(bootStep - 1); });
    player.querySelector('[data-boot="next"]').addEventListener("click", () => { stopBootPlayback(); setBootStep(bootStep + 1); });
    player.querySelector('[data-boot="toggle"]').addEventListener("click", () => bootPlaying ? stopBootPlayback() : startBootPlayback());
    player.querySelectorAll("[data-boot-step]").forEach(button => button.addEventListener("click", () => { stopBootPlayback(); setBootStep(Number(button.dataset.bootStep)); }));
    applyBootStep();
    if (!matchMedia("(prefers-reduced-motion: reduce)").matches) startBootPlayback();
  }

  function expandedBlockIds(ids) {
    const expanded = new Set(ids);
    ids.forEach(id => ancestorsOf(id).forEach(block => expanded.add(block.id)));
    return expanded;
  }

  function applyBootStep() {
    if (!activeBootBehavior || !document.getElementById("boot-player")) return;
    const step = activeBootBehavior.steps[bootStep];
    const active = new Set(step.blocks);
    const activeWithContext = expandedBlockIds(active);
    const visited = new Set(activeBootBehavior.steps.slice(0, bootStep + 1).flatMap(item => item.blocks));
    const visitedWithContext = expandedBlockIds(visited);
    document.querySelectorAll(".diagram-node, .internal-node").forEach(node => {
      const id = node.dataset.id || node.dataset.child;
      node.classList.toggle("boot-active", active.has(id));
      node.classList.toggle("boot-context", !active.has(id) && activeWithContext.has(id));
      node.classList.toggle("boot-visited", !activeWithContext.has(id) && visited.has(id));
      node.classList.toggle("boot-muted", !activeWithContext.has(id) && !visited.has(id));
    });
    document.querySelectorAll(".edge, .internal-edge").forEach(edge => {
      const from = edge.dataset.from;
      const to = edge.dataset.to;
      const live = activeWithContext.has(from) || activeWithContext.has(to);
      const traversed = visitedWithContext.has(from) && visitedWithContext.has(to);
      edge.classList.toggle("boot-live", live);
      edge.classList.toggle("boot-traversed", !live && traversed);
      edge.classList.toggle("boot-dormant", !live && !traversed);
    });
    document.querySelectorAll(".edge-label").forEach(label => {
      const edge = document.querySelector(`.edge[data-connection="${CSS.escape(label.dataset.connection)}"]`);
      label.classList.toggle("boot-label-live", Boolean(edge?.classList.contains("boot-live")));
      label.classList.toggle("boot-label-dormant", Boolean(edge?.classList.contains("boot-dormant")));
    });
    document.getElementById("boot-step-count").textContent = `Step ${bootStep + 1} of ${activeBootBehavior.steps.length}`;
    document.getElementById("boot-step-mode").textContent = step.parallel ? "Concurrent hardware" : (step.branch ? "Alternative path" : "Ordered");
    document.getElementById("boot-step-title").textContent = step.title;
    document.getElementById("boot-step-detail").textContent = step.detail;
    document.getElementById("boot-step-blocks").innerHTML = step.blocks.map(id => `<button type="button" data-block="${escapeHtml(id)}">${escapeHtml(byId.get(id)?.name || id)}</button>`).join("");
    document.querySelectorAll("#boot-step-blocks [data-block]").forEach(button => button.addEventListener("click", () => selectBlock(button.dataset.block, { openSource: true })));
    document.querySelectorAll("[data-boot-step]").forEach((button, index) => {
      button.classList.toggle("complete", index < bootStep);
      button.classList.toggle("current", index === bootStep);
      if (index === bootStep) button.setAttribute("aria-current", "step"); else button.removeAttribute("aria-current");
    });
    document.querySelector('[data-boot="previous"]').disabled = bootStep === 0;
    document.querySelector('[data-boot="next"]').disabled = bootStep === activeBootBehavior.steps.length - 1;
    document.getElementById("boot-announcement").textContent = `${document.getElementById("boot-step-count").textContent}: ${step.title}. ${step.detail}`;
    updateBootToggle();
  }

  function setBootStep(index) {
    if (!activeBootBehavior) return;
    bootStep = Math.max(0, Math.min(activeBootBehavior.steps.length - 1, index));
    applyBootStep();
  }

  function updateBootToggle() {
    const toggle = document.querySelector('[data-boot="toggle"]');
    if (!toggle) return;
    toggle.innerHTML = bootPlaying ? `${icons.pause}<span>Pause</span>` : `${icons.play}<span>Play</span>`;
    toggle.setAttribute("aria-label", bootPlaying ? "Pause boot animation" : "Play boot animation");
  }

  function startBootPlayback() {
    if (!activeBootBehavior || bootPlaying) return;
    if (bootStep === activeBootBehavior.steps.length - 1) bootStep = 0;
    bootPlaying = true;
    applyBootStep();
    bootTimer = window.setInterval(() => {
      if (bootStep >= activeBootBehavior.steps.length - 1) { stopBootPlayback(); return; }
      setBootStep(bootStep + 1);
    }, 2300);
  }

  function stopBootPlayback() {
    if (bootTimer !== null) window.clearInterval(bootTimer);
    bootTimer = null;
    bootPlaying = false;
    updateBootToggle();
  }

  function renderBoot() {
    const behavior = model.behaviors.find(item => item.id === "boot") || model.behaviors[0];
    renderStructure({ bootBehavior: behavior, focusId: model.product.default_focus });
    els.breadcrumbs.innerHTML = `<button class="crumb" type="button" data-boot-structure>${escapeHtml(byId.get(model.product.default_focus)?.name || "Coral SoC")}</button><span class="crumb-separator">/</span><span class="crumb current" aria-current="page">Boot flow</span>`;
    els.breadcrumbs.querySelector("[data-boot-structure]").addEventListener("click", () => setView("structure"));
    els.model_count.textContent = `${behavior.steps.length} animated boot steps`;
    requestAnimationFrame(fitDiagram);
  }

  function renderOwnership() {
    document.getElementById("boot-player")?.remove();
    document.querySelector(".canvas-panel")?.classList.remove("boot-mode");
    activeBootBehavior = null;
    const blocks = model.blocks.filter(block => block.id !== "coral-system");
    els.diagram_viewport.className = "diagram-viewport ownership-stage";
    els.diagram_viewport.innerHTML = `<header class="ownership-heading"><h1>Ownership is architecture</h1><p>Unassigned work is visible by design. Assign owners and maturity in the model; keep detailed execution in linked issues.</p></header><table class="ownership-table"><thead><tr><th>Block</th><th>Owner</th><th>Maturity</th><th>Evidence</th><th>Issues</th></tr></thead><tbody>${blocks.map(block => `<tr><td><button type="button" data-block="${escapeHtml(block.id)}">${escapeHtml(block.name)}</button></td><td>${escapeHtml(owners.get(block.owner)?.name || block.owner)}</td><td><span class="state-chip" style="--state:${maturityColors[block.maturity]}">${escapeHtml(stateLabel(block.maturity))}</span></td><td>${block.documents.length} doc${block.documents.length === 1 ? "" : "s"}</td><td>${block.issues.length ? block.issues.length : "None linked"}</td></tr>`).join("")}</tbody></table>`;
    els.diagram_viewport.querySelectorAll("[data-block]").forEach(button => button.addEventListener("click", () => selectBlock(button.dataset.block, { openSource: true })));
    renderBreadcrumbs("coral-system");
    els.model_count.textContent = `${blocks.filter(block => block.owner === "unassigned").length} unassigned · ${blocks.length} total`;
    els.protocol_legend.innerHTML = Object.entries(maturityColors).map(([stateName, color]) => `<span class="legend-item"><span class="legend-line" style="--legend:${color}"></span>${escapeHtml(stateLabel(stateName))}</span>`).join("");
    els.protocol_legend.hidden = false;
  }

  function documentHref(documentRef) {
    if (documentRef.url) return documentRef.url;
    let href = `${documentRef.path}`;
    if (documentRef.page) href += `#page=${documentRef.page}`;
    else if (documentRef.anchor) href += `#${documentRef.anchor}`;
    return href;
  }

  function renderInspector(block) {
    els.inspector_empty.hidden = true;
    els.inspector_content.hidden = false;
    const related = model.connections.filter(connection => connection.from === block.id || connection.to === block.id);
    els.inspector_content.innerHTML = `<div class="inspector-kicker"><span>${escapeHtml(block.kind.replaceAll("-", " "))}</span><span>${related.length} interface${related.length === 1 ? "" : "s"}</span></div><h1>${escapeHtml(block.name)}</h1><p class="inspector-summary">${escapeHtml(block.summary)}</p><div class="status-strip"><div class="status-cell"><span>Owner</span><strong>${escapeHtml(owners.get(block.owner)?.name || block.owner)}</strong></div><div class="status-cell"><span>Maturity</span><strong class="maturity-dot" style="--state:${maturityColors[block.maturity]}">${escapeHtml(stateLabel(block.maturity))}</strong></div></div><section class="inspector-section"><h2>Responsibilities</h2><ul class="responsibility-list">${block.responsibilities.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul></section><section class="inspector-section"><h2>Documents</h2><div class="resource-list">${block.documents.length ? block.documents.map(documentRef => `<a class="resource-link" href="${escapeHtml(documentHref(documentRef))}" data-local-link="${escapeHtml(documentRef.path || "")}"${documentRef.url ? ' target="_blank" rel="noreferrer"' : ""}><span class="resource-icon">${documentRef.url ? icons.external : icons.file}</span><span class="resource-copy"><strong>${escapeHtml(documentRef.title)}</strong><small>${escapeHtml(documentRef.path || documentRef.url)}${documentRef.page ? ` · page ${documentRef.page}` : ""}</small></span><span class="resource-arrow">→</span></a>`).join("") : '<p class="empty-note">No documents linked yet.</p>'}</div></section><section class="inspector-section"><h2>Issues</h2>${block.issues.length ? `<div class="resource-list">${block.issues.map(issue => `<a class="resource-link" href="${escapeHtml(issue.url)}" target="_blank" rel="noreferrer"><span class="resource-icon">${icons.external}</span><span class="resource-copy"><strong>${escapeHtml(issue.title || "Linked issue")}</strong><small>${escapeHtml(issue.url)}</small></span><span class="resource-arrow">→</span></a>`).join("")}</div>` : '<p class="empty-note">No linked issue yet. Add the issue URL when work is scheduled.</p>'}</section><section class="inspector-section"><h2>RTL references</h2>${block.rtl.length ? `<div class="resource-list">${block.rtl.map(ref => `<a class="resource-link" href="${escapeHtml(ref.url)}" target="_blank" rel="noreferrer"><span class="resource-icon">${icons.external}</span><span class="resource-copy"><strong>${escapeHtml(ref.symbol || ref.path)}</strong><small>${escapeHtml(ref.repository)} · ${escapeHtml(ref.revision)}</small></span><span class="resource-arrow">→</span></a>`).join("")}</div>` : '<p class="empty-note">No confirmed RTL anchor yet. Add the repository, revision, path, and declaration when implementation begins.</p>'}</section>${hasChildren(block.id) ? `<button class="enter-block" type="button" data-enter="${escapeHtml(block.id)}">Open ${escapeHtml(block.name)} internals →</button>` : ""}`;
    els.inspector_content.querySelector("[data-enter]")?.addEventListener("click", event => enterBlock(event.currentTarget.dataset.enter));
    els.inspector_content.querySelectorAll("[data-local-link]").forEach(link => link.addEventListener("click", event => {
      if (!vscode || !link.dataset.localLink) return;
      event.preventDefault();
      vscode.postMessage({ type: "openFile", file: link.dataset.localLink, href: link.getAttribute("href") });
    }));
  }

  function render() {
    document.querySelectorAll("[data-view]").forEach(button => {
      const active = button.dataset.view === state.view;
      button.classList.toggle("active", active);
      if (button.matches("[role=tab]")) button.setAttribute("aria-selected", String(active));
    });
    if (state.view === "boot") renderBoot();
    else if (state.view === "protocols") renderProtocols();
    else if (state.view === "instructions") renderInstructions();
    else if (state.view === "ownership") renderOwnership();
    else renderStructure();
    if (state.selected && byId.has(state.selected)) renderInspector(byId.get(state.selected));
    else { els.inspector_empty.hidden = false; els.inspector_content.hidden = true; }
    els.zoom_value.textContent = `${Math.round(state.zoom * 100)}%`;
    document.querySelector(".zoom-dock").hidden = !["structure", "boot"].includes(state.view);
    updateArrangeUi();
    updateHierarchyNavigation();
  }

  function openSearch() {
    searchOpener = document.activeElement;
    els.search_layer.hidden = false;
    els.search_input.value = state.query;
    renderSearchResults();
    requestAnimationFrame(() => els.search_input.focus());
  }

  function closeSearch() { els.search_layer.hidden = true; (searchOpener instanceof HTMLElement ? searchOpener : document.getElementById("search-trigger")).focus(); }

  function renderSearchResults() {
    const query = els.search_input.value.trim().toLowerCase();
    state.query = query;
    const matches = model.blocks.filter(block => {
      const haystack = [block.name, block.kind, block.summary, ...block.responsibilities, ...block.documents.flatMap(item => [item.title, item.path])].join(" ").toLowerCase();
      return (!query || haystack.includes(query)) && (!state.status || block.maturity === state.status) && (!state.owner || block.owner === state.owner);
    });
    els.search_results.innerHTML = matches.length ? matches.map(block => `<button class="search-result" type="button" data-result="${escapeHtml(block.id)}"><strong>${escapeHtml(block.name)}</strong><small>${escapeHtml(block.kind)}</small><span>${escapeHtml(block.summary)}</span></button>`).join("") : '<p class="empty-note">No blocks match that search. Try a protocol, document, or shorter term.</p>';
    els.search_results.querySelectorAll("[data-result]").forEach(button => button.addEventListener("click", () => {
      const id = button.dataset.result;
      const parent = byId.get(id)?.parent;
      if (parent) state.focus = parent;
      else if (hasChildren(id)) state.focus = id;
      state.view = "structure";
      state.selected = id;
      persist();
      closeSearch();
      render();
    }));
  }

  function setZoom(value) {
    const minimum = window.innerWidth < 600 ? .8 : .6;
    state.zoom = Math.max(minimum, Math.min(1.5, Math.round(value * 20) / 20));
    els.diagram_stage?.style.setProperty("--zoom", state.zoom);
    syncStageBounds();
    els.zoom_value.textContent = `${Math.round(state.zoom * 100)}%`;
  }

  function syncStageBounds() {
    if (!els.diagram_stage || !els.diagram_stage_shell) return;
    const scaledWidth = els.diagram_stage.offsetWidth * state.zoom;
    const scaledHeight = els.diagram_stage.offsetHeight * state.zoom;
    const panMargin = 180;
    const shellWidth = Math.max(els.diagram_viewport.clientWidth, scaledWidth) + panMargin * 2;
    const shellHeight = Math.max(els.diagram_viewport.clientHeight, scaledHeight) + panMargin * 2;
    els.diagram_stage_shell.style.width = `${shellWidth}px`;
    els.diagram_stage_shell.style.height = `${shellHeight}px`;
    els.diagram_stage.style.left = `${Math.max(0, (shellWidth - scaledWidth) / 2)}px`;
    els.diagram_stage.style.top = `${Math.max(0, (shellHeight - scaledHeight) / 2)}px`;
  }

  function centerDiagram(behavior = "smooth") {
    els.diagram_viewport.scrollTo({
      left: Math.max(0, (els.diagram_viewport.scrollWidth - els.diagram_viewport.clientWidth) / 2),
      top: Math.max(0, (els.diagram_viewport.scrollHeight - els.diagram_viewport.clientHeight) / 2),
      behavior
    });
  }

  function fitDiagram() {
    if (!els.diagram_stage) return;
    const availableWidth = els.diagram_viewport.clientWidth - 32;
    const availableHeight = els.diagram_viewport.clientHeight - 32;
    setZoom(Math.min(1, availableWidth / els.diagram_stage.offsetWidth, availableHeight / els.diagram_stage.offsetHeight));
    requestAnimationFrame(() => centerDiagram());
  }

  function updateArrangeUi() {
    const available = state.view === "structure" && Boolean(activeStructure);
    els.arrange_toggle.hidden = !available;
    els.arrange_actions.hidden = !available || !state.arrange;
    els.arrange_toggle.classList.toggle("active", available && state.arrange);
    els.arrange_toggle.setAttribute("aria-pressed", String(available && state.arrange));
    document.querySelector(".canvas-panel")?.classList.toggle("arrange-mode", available && state.arrange);
    els.diagram_viewport.classList.toggle("arrange-active", available && state.arrange);
    document.getElementById("reset-positions").disabled = !layoutEdits[state.focus] || !Object.keys(layoutEdits[state.focus]).length;
  }

  function setArrange(enabled) {
    if (state.view !== "structure") return;
    state.arrange = enabled;
    updateArrangeUi();
    document.querySelectorAll(".diagram-node").forEach(node => {
      const block = byId.get(node.dataset.id);
      if (block) node.setAttribute("aria-label", `${block.name}, ${block.kind}, ${stateLabel(block.maturity)}${hasChildren(block.id) ? `, contains ${childrenOf(block.id).length} blocks` : ""}${enabled ? ", movable with arrow keys" : ""}`);
    });
    showToast(enabled ? "Arrange mode on — drag blocks or use arrow keys." : "Arrange mode off.");
  }

  function moveBlock(id, x, y, commit = false) {
    if (!activeStructure) return;
    const position = activeStructure.layout.positions.get(id);
    const node = els.node_layer.querySelector(`.diagram-node[data-id="${CSS.escape(id)}"]`);
    if (!position || !node) return;
    position.x = Math.round(Math.max(8, Math.min(activeStructure.layout.width - position.w - 8, x)));
    position.y = Math.round(Math.max(8, Math.min(activeStructure.layout.height - position.h - 8, y)));
    node.style.left = `${position.x}px`;
    node.style.top = `${position.y}px`;
    refreshStructureGeometry();
    if (!commit) return;
    layoutEdits[activeStructure.focus.id] ||= {};
    layoutEdits[activeStructure.focus.id][id] = { x: position.x, y: position.y };
    persist();
    updateArrangeUi();
  }

  function resetPositions() {
    if (!layoutEdits[state.focus]) return;
    delete layoutEdits[state.focus];
    persist();
    render();
    requestAnimationFrame(() => fitDiagram());
    showToast(`Restored authored positions for ${byId.get(state.focus)?.name || state.focus}.`);
  }

  function positionsAsYaml() {
    const authored = model.layouts[state.focus];
    if (!authored) return "";
    const lines = [
      `  ${state.focus}:`,
      `    canvas: {width: ${authored.canvas.width}, height: ${authored.canvas.height}}`,
      "    nodes:"
    ];
    Object.entries(authored.nodes).forEach(([id, source]) => {
      const position = activeStructure?.layout.positions.get(id) || source;
      const width = source.width;
      const height = source.height;
      lines.push(`      ${id}: {x: ${Math.round(position.x)}, y: ${Math.round(position.y)}, width: ${width}, height: ${height}}`);
    });
    return lines.join("\n");
  }

  async function copyPositions() {
    const yaml = positionsAsYaml();
    if (!yaml) return;
    if (vscode) vscode.postMessage({ type: "copyText", text: yaml });
    else {
      try { await navigator.clipboard.writeText(yaml); }
      catch {
        const field = document.createElement("textarea");
        field.value = yaml;
        document.body.appendChild(field);
        field.select();
        document.execCommand("copy");
        field.remove();
      }
    }
    showToast("Copied current layout as YAML.");
  }

  function beginCanvasGesture(event) {
    if (!["structure", "boot"].includes(state.view) || ![0, 1].includes(event.button)) return;
    const node = event.target.closest(".diagram-node");
    const forcePan = spacePressed || event.button === 1;
    if (state.arrange && node && !forcePan && event.button === 0) {
      const position = activeStructure?.layout.positions.get(node.dataset.id);
      if (!position) return;
      activeGesture = { type: "move", pointerId: event.pointerId, node, id: node.dataset.id, startX: event.clientX, startY: event.clientY, x: position.x, y: position.y, moved: false };
      node.focus({ preventScroll: true });
    } else {
      const pending = Boolean(node && event.button === 0 && !forcePan);
      activeGesture = { type: "pan", pointerId: event.pointerId, startX: event.clientX, startY: event.clientY, left: els.diagram_viewport.scrollLeft, top: els.diagram_viewport.scrollTop, moved: false, pending };
      if (!pending) els.diagram_viewport.classList.add("is-panning");
    }
    if (!activeGesture.pending) {
      els.diagram_viewport.setPointerCapture(event.pointerId);
      event.preventDefault();
    }
  }

  function continueCanvasGesture(event) {
    if (!activeGesture || activeGesture.pointerId !== event.pointerId) return;
    const dx = event.clientX - activeGesture.startX;
    const dy = event.clientY - activeGesture.startY;
    if (Math.hypot(dx, dy) > 3) {
      activeGesture.moved = true;
      if (activeGesture.pending) {
        activeGesture.pending = false;
        els.diagram_viewport.classList.add("is-panning");
        els.diagram_viewport.setPointerCapture(event.pointerId);
      }
    }
    if (activeGesture.type === "pan") {
      if (activeGesture.pending) return;
      els.diagram_viewport.scrollLeft = activeGesture.left - dx;
      els.diagram_viewport.scrollTop = activeGesture.top - dy;
      event.preventDefault();
    } else {
      activeGesture.node.classList.add("dragging");
      moveBlock(activeGesture.id, activeGesture.x + dx / state.zoom, activeGesture.y + dy / state.zoom);
    }
  }

  function endCanvasGesture(event) {
    if (!activeGesture || activeGesture.pointerId !== event.pointerId) return;
    const gesture = activeGesture;
    activeGesture = null;
    els.diagram_viewport.classList.remove("is-panning");
    if (els.diagram_viewport.hasPointerCapture(event.pointerId)) els.diagram_viewport.releasePointerCapture(event.pointerId);
    if (gesture.type === "move") {
      gesture.node.classList.remove("dragging");
      if (gesture.moved) {
        const position = activeStructure?.layout.positions.get(gesture.id);
        if (position) moveBlock(gesture.id, position.x, position.y, true);
        suppressNextClick = true;
        showToast(`Moved ${byId.get(gesture.id)?.name || gesture.id}.`);
      }
    } else if (gesture.moved) {
      suppressNextClick = true;
    }
  }

  function panCanvasBy(left, top, behavior = "auto") {
    els.diagram_viewport.scrollBy({ left, top, behavior });
  }

  function panCanvasWheel(event) {
    if (!["structure", "boot"].includes(state.view)) return;
    const scale = event.deltaMode === WheelEvent.DOM_DELTA_LINE ? 18 : event.deltaMode === WheelEvent.DOM_DELTA_PAGE ? els.diagram_viewport.clientHeight : 1;
    const vertical = event.deltaY * scale;
    const horizontal = event.deltaX * scale + (event.shiftKey && !event.deltaX ? vertical : 0);
    panCanvasBy(horizontal, event.shiftKey && !event.deltaX ? 0 : vertical);
    event.preventDefault();
  }

  function exportDiagram(format) {
    if (state.view !== "structure") { showToast("Export is available from the structure view."); return; }
    const svg = buildExportSvg();
    if (format === "svg") downloadBlob(new Blob([svg], { type: "image/svg+xml" }), `coral-${state.focus}.svg`);
    else {
      const image = new Image();
      const blob = new Blob([svg], { type: "image/svg+xml" });
      const url = URL.createObjectURL(blob);
      image.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = image.width * 2;
        canvas.height = image.height * 2;
        const context = canvas.getContext("2d");
        context.scale(2, 2);
        context.drawImage(image, 0, 0);
        canvas.toBlob(png => downloadBlob(png, `coral-${state.focus}.png`), "image/png");
        URL.revokeObjectURL(url);
      };
      image.src = url;
    }
  }

  function buildExportSvg() {
    const stage = els.diagram_stage;
    const edgeMarkup = els.edge_layer?.innerHTML || "";
    const nodes = [...document.querySelectorAll(".diagram-node")].map(node => {
      const x = parseFloat(node.style.left), y = parseFloat(node.style.top), width = node.offsetWidth, height = node.offsetHeight;
      const block = byId.get(node.dataset.id);
      const fill = block.kind === "external-memory" ? "#c25700" : "#101c26";
      return `<g><rect x="${x}" y="${y}" width="${width}" height="${height}" rx="5" fill="${fill}" stroke="#101c26"/><rect x="${x}" y="${y}" width="${width}" height="6" rx="2" fill="${kindColors[block.kind] || "#64748b"}"/><text x="${x + 14}" y="${y + 40}" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#ffffff">${escapeHtml(block.name)}</text><text x="${x + 14}" y="${y + 62}" font-family="monospace" font-size="9" fill="#c8d4da">${escapeHtml(block.kind.toUpperCase())}</text></g>`;
    }).join("");
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${stage.offsetWidth}" height="${stage.offsetHeight}" viewBox="0 0 ${stage.offsetWidth} ${stage.offsetHeight}"><rect width="100%" height="100%" fill="#f7f9fa"/><g>${edgeMarkup}</g>${nodes}</svg>`;
  }

  function downloadBlob(blob, name) {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
    showToast(`Exported ${name}`);
  }

  function showToast(message) {
    clearTimeout(toastTimer);
    els.toast.textContent = message;
    els.toast.classList.add("visible");
    toastTimer = setTimeout(() => els.toast.classList.remove("visible"), 2600);
  }

  function populateFilters() {
    [...new Set(model.blocks.map(block => block.maturity))].sort().forEach(value => els.status_filter.add(new Option(stateLabel(value), value)));
    model.owners.forEach(owner => els.owner_filter.add(new Option(owner.name, owner.id)));
  }

  document.querySelectorAll("[data-view]").forEach(button => button.addEventListener("click", () => setView(button.dataset.view)));
  document.querySelectorAll(".view-tab").forEach((tab, index, tabs) => tab.addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const nextIndex = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    tabs[nextIndex].focus();
    setView(tabs[nextIndex].dataset.view);
  }));
  document.getElementById("search-trigger").addEventListener("click", openSearch);
  document.getElementById("search-close").addEventListener("click", closeSearch);
  document.querySelector(".search-backdrop").addEventListener("click", closeSearch);
  els.search_input.addEventListener("input", renderSearchResults);
  els.status_filter.addEventListener("change", event => { state.status = event.target.value; renderSearchResults(); });
  els.owner_filter.addEventListener("change", event => { state.owner = event.target.value; renderSearchResults(); });
  document.getElementById("zoom-in").addEventListener("click", () => setZoom(state.zoom + .1));
  document.getElementById("zoom-out").addEventListener("click", () => setZoom(state.zoom - .1));
  document.getElementById("zoom-fit").addEventListener("click", fitDiagram);
  els.arrange_toggle.addEventListener("click", () => setArrange(!state.arrange));
  els.nav_up.addEventListener("click", focusParent);
  els.nav_down.addEventListener("click", () => { if (state.selected && hasChildren(state.selected)) enterBlock(state.selected); });
  document.getElementById("copy-positions").addEventListener("click", copyPositions);
  document.getElementById("reset-positions").addEventListener("click", resetPositions);
  document.getElementById("export-svg").addEventListener("click", () => exportDiagram("svg"));
  document.getElementById("export-png").addEventListener("click", () => exportDiagram("png"));
  els.diagram_viewport.addEventListener("pointerdown", beginCanvasGesture);
  els.diagram_viewport.addEventListener("pointermove", continueCanvasGesture);
  els.diagram_viewport.addEventListener("pointerup", endCanvasGesture);
  els.diagram_viewport.addEventListener("pointercancel", endCanvasGesture);
  els.diagram_viewport.addEventListener("wheel", panCanvasWheel, { passive: false });
  els.diagram_viewport.addEventListener("click", event => {
    if (!suppressNextClick) return;
    suppressNextClick = false;
    event.preventDefault();
    event.stopImmediatePropagation();
  }, true);
  document.addEventListener("keydown", event => {
    const editingText = ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName);
    if (event.code === "Space" && !editingText && !document.activeElement.matches("button, a")) {
      spacePressed = true;
      els.diagram_viewport.classList.add("force-pan");
      event.preventDefault();
    }
    if (state.arrange && !event.altKey && !editingText && ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key) && document.activeElement.classList.contains("diagram-node")) {
      const node = document.activeElement;
      const position = activeStructure?.layout.positions.get(node.dataset.id);
      if (position) {
        event.preventDefault();
        const amount = event.shiftKey ? 1 : 10;
        const dx = event.key === "ArrowLeft" ? -amount : event.key === "ArrowRight" ? amount : 0;
        const dy = event.key === "ArrowUp" ? -amount : event.key === "ArrowDown" ? amount : 0;
        moveBlock(node.dataset.id, position.x + dx, position.y + dy, true);
        const moved = activeStructure.layout.positions.get(node.dataset.id);
        showToast(`${byId.get(node.dataset.id)?.name || node.dataset.id}: x ${moved.x}, y ${moved.y}.`);
      }
    }
    if (!["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName) && document.activeElement === els.diagram_viewport && !event.altKey && ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "PageUp", "PageDown"].includes(event.key)) {
      event.preventDefault();
      const amount = event.shiftKey ? 24 : 88;
      const left = event.key === "ArrowLeft" ? -amount : event.key === "ArrowRight" ? amount : 0;
      const top = event.key === "ArrowUp" ? -amount : event.key === "ArrowDown" ? amount : event.key === "PageUp" ? -els.diagram_viewport.clientHeight * .8 : event.key === "PageDown" ? els.diagram_viewport.clientHeight * .8 : 0;
      panCanvasBy(left, top, "smooth");
    }
    if (event.key === "/" && !["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) { event.preventDefault(); openSearch(); }
    if (event.key === "Escape" && !els.search_layer.hidden) closeSearch();
    else if (event.key === "Escape" && state.arrange) setArrange(false);
    if (state.view === "structure" && event.altKey && event.key === "ArrowLeft" && byId.get(state.focus)?.parent) { event.preventDefault(); focusParent(); }
    if (state.view === "structure" && event.altKey && event.key === "ArrowUp" && byId.get(state.focus)?.parent) { event.preventDefault(); focusParent(); }
    if (state.view === "structure" && event.altKey && event.key === "ArrowDown" && state.selected && hasChildren(state.selected)) { event.preventDefault(); enterBlock(state.selected); }
    if (event.key === "Tab" && !els.search_layer.hidden) {
      const focusable = [...els.search_layer.querySelectorAll('button:not([disabled]), input:not([disabled]), select:not([disabled]), a[href]')].filter(item => item.offsetParent !== null);
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
  });
  document.addEventListener("keyup", event => {
    if (event.code !== "Space") return;
    spacePressed = false;
    els.diagram_viewport.classList.remove("force-pan");
  });
  document.addEventListener("visibilitychange", () => { if (document.hidden) stopBootPlayback(); });
  window.addEventListener("blur", () => { spacePressed = false; els.diagram_viewport.classList.remove("force-pan"); });
  window.addEventListener("resize", () => { syncStageBounds(); });

  window.addEventListener("message", event => {
    const message = event.data;
    if (message.type === "editorSync") {
      els.sync_indicator.textContent = message.label || "Source synchronized";
      if (message.symbolId && byId.has(message.symbolId)) {
        const block = byId.get(message.symbolId);
        state.view = "structure";
        state.focus = block.parent || model.product.default_focus;
        state.selected = block.id;
        persist();
        render();
        requestAnimationFrame(() => document.querySelector(`[data-id="${CSS.escape(block.id)}"]`)?.classList.add("highlighted"));
      }
    }
  });

  populateFilters();
  render();
  vscode?.postMessage({ type: "ready" });
})();
