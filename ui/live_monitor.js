// Live UI monitor: observes #realtimeResult and #realtimeConfirmed and POSTs
// the latest snapshot to /api/debug/state every 200ms.  Used by the server
// developer to inspect page state via curl.  No-op on pages that don't have
// the targets (e.g. the unit-test harness).
(function () {
  "use strict";
  if (window.__liveMonitorInstalled) return;
  window.__liveMonitorInstalled = true;

  const ENDPOINT = "/api/debug/state";
  const LOG = [];
  const MAX_LOG = 50;

  function snap(id) {
    const el = document.getElementById(id);
    if (!el) return null;
    return {
      id: id,
      text: (el.textContent || "").slice(0, 300),
      inner: (el.innerHTML || "").slice(0, 500),
    };
  }
  function snapAll() {
    return {
      ts: Date.now(),
      live: snap("realtimeResult"),
      archive: snap("realtimeConfirmed"),
      status: snap("realtimeStatus"),
      health: snap("healthBadge"),
      log: LOG.slice(-20),
    };
  }
  function post() {
    try {
      fetch(ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(snapAll()),
        keepalive: true,
      }).catch(() => {});
    } catch (_e) {}
  }
  function attach(id) {
    const el = document.getElementById(id);
    if (!el) return;
    const ob = new MutationObserver((muts) => {
      for (const m of muts) {
        LOG.push({
          t: Math.round(performance.now()),
          target: id,
          kind: m.type,
          added: m.addedNodes.length,
          removed: m.removedNodes.length,
        });
      }
      while (LOG.length > MAX_LOG) LOG.shift();
    });
    ob.observe(el, { childList: true, subtree: true, characterData: true, attributes: true });
  }
  function install() {
    attach("realtimeResult");
    attach("realtimeConfirmed");
    attach("realtimeStatus");
    /* Reduced from 200ms to 1000ms: 5/s → 1/s, saving ~4 req/s per
     * page.  The monitor is for human inspection via curl, not
     * control-loop, so 1Hz is plenty. */
    setInterval(post, 1000);
    // also post on key DOM changes
    document.addEventListener("DOMContentLoaded", post);
    setTimeout(post, 50);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", install);
  } else {
    install();
  }
})();
