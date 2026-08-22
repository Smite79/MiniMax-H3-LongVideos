// Write auto-derived flow shifts back into the widgets.
//
// A node cannot set a widget from Python -- widgets are frontend state, and run()
// only ever RECEIVES them. Without this, auto_shift left the graph lying: the
// widget still read 12/3 while the render used 1.89/0.47, so a saved workflow did
// not describe the render that produced it.
//
// sampler.py returns {"ui": {"h3_shift": [video, audio]}, "result": ...} whenever
// it changed them; comfy/execution.py delivers that dict to the "executed" event.
// Nothing is sent when the values were not changed, so a graph the user controls
// is never touched.

// THREE levels up, not two. server.py:363-366 serves this at
// /extensions/H3-LongVideos-V1/js/autoshift.js -- the WEB_DIRECTORY is the
// /extensions/<name>/ root, so a file in its js/ subfolder sits three deep.
// At ../../ these resolved to /extensions/scripts/app.js, a 404 that silently
// aborted the whole module, so no listener was ever registered and the shifts
// were never written back.
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_TYPES = new Set([
    "H3LongVideos",
    "H3LongVideosFL2VA",
    "H3LongVideosV1",
    "H3LongVideosREF2VA",
]);

function setWidget(node, name, value) {
    const w = node.widgets?.find((x) => x.name === name);
    // Only touch a real number, and only when it actually differs -- assigning an
    // identical value still marks the graph dirty, which would prompt to save a
    // workflow that did not change.
    if (!w || typeof value !== "number" || !isFinite(value)) return false;
    if (Math.abs((w.value ?? 0) - value) < 1e-9) return false;
    w.value = value;
    w.callback?.(value);
    return true;
}

app.registerExtension({
    name: "H3LongVideos.autoshift",
    setup() {
        api.addEventListener("executed", ({ detail }) => {
            const payload = detail?.output?.h3_shift;
            if (!Array.isArray(payload) || payload.length < 2) return;

            const node = app.graph?.getNodeById(detail.node);
            if (!node || !NODE_TYPES.has(node.type)) return;

            const [video, audio] = payload.map(Number);
            const a = setWidget(node, "shift_video", video);
            const b = setWidget(node, "shift_audio", audio);
            if (!a && !b) return;

            // The node records what it wrote, so a widget still holding that value
            // is recognised as ITS OWN on the next run and re-derived if `steps`
            // changes. Only a value the user types stops it. Without that record,
            // writing back would disable the feature after a single run.
            console.log(
                `[H3 Long Videos] auto_shift set shift_video=${video}, ` +
                `shift_audio=${audio} to match the step count. Type a shift ` +
                `yourself to take over.`
            );
            node.setDirtyCanvas(true, true);
        });
    },
});
