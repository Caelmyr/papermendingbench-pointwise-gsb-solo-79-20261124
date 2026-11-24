"""Routes for the paper mending workbench and archived reading evidence."""
from io import BytesIO

from flask import Blueprint, Response, jsonify, render_template, request, send_file

from image_lab.validation import Conflict, Invalid

from . import analysis, demo, evidence, photos
from .model import MARKS, READINGS, ROLES, choice, locate, mark_fields, obj, run_fields, text
from .store import Library, identity, stamp

MEASUREMENTS = {
    "sheet-edge": (analysis.sheet_edge, {"region": "sheet"}, {}),
    "loss-map": (analysis.loss_map, {"region": "loss"}, {}),
    "tear-section": (analysis.tear_section, {"line": "tear"}, {}),
    "patch-coverage": (analysis.patch_coverage, {"losses": "loss", "patch": "patch"}, {}),
    "fiber-balance": (analysis.fiber_balance, {"region": "fiber"}, {}),
    "ink-spread": (analysis.ink_spread, {"line": "ink"}, {}),
    "foxing-map": (analysis.foxing_map, {"region": "stain"}, {}),
    "paper-tone": (analysis.paper_tone, {"region": "inspect"}, {}),
    "landmark-offset": (analysis.landmark_offset, {"reference": "register"}, {"comparison": "register"}),
    "fold-alignment": (analysis.fold_alignment,
                       {"region": "inspect", "fold": "fold", "reference": "register"},
                       {"comparison": "register"}),
    "paired-support": (analysis.paired_support, {"region": "inspect"}, {}),
    "transmission-change": (analysis.transmission_difference, {"region": "inspect"}, {}),
    "watermark-contrast": (analysis.watermark_contrast, {"watermark": "watermark", "reference": "control"}, {}),
    "fold-ligaments": (analysis.fold_ligaments, {"region": "inspect", "fold": "fold"}, {}),
}
PAIRED = {"landmark-offset", "paired-support", "transmission-change", "fold-alignment"}
PAIRED.update(key for key in analysis.ALIGNMENT_MODE_ALIASES if key != "fold-alignment")
for alias in analysis.ALIGNMENT_MODE_ALIASES:
    MEASUREMENTS.setdefault(alias, MEASUREMENTS["fold-alignment"])
HIDDEN_MODES = set(analysis.ALIGNMENT_MODE_ALIASES) - {"fold-alignment"}
for alias in HIDDEN_MODES:
    READINGS[alias] = READINGS["fold-alignment"]
OPTIONAL_ROLE_SLOTS = {
    mode: {"region", "fold", "reference", "comparison", "other_fold"}
    for mode in analysis.ALIGNMENT_MODE_ALIASES
}
OPTIONS = {
    "sheet-edge": {"luma": ["Sheet luminance", .47, .01]},
    "loss-map": {"transmission": ["Loss transmission", .78, .01],
                 "minimum_mm2": ["Minimum loss area (mm²)", .3, .01]},
    "tear-section": {"transmission": ["Tear transmission", .78, .01]},
    "patch-coverage": {"transmission": ["Loss transmission", .78, .01]},
    "ink-spread": {"chroma": ["Ink chroma distance", .08, .01]},
    "foxing-map": {"yellow": ["Local yellow excess", .035, .005]},
    "transmission-change": {"difference": ["Transmission difference", .12, .01]},
    "fold-alignment": {
        "max_residual_mm": ["Landmark residual cap (mm)", 1, .05],
        "min_common_mm2": ["Minimum deduplicated common area (mm²)", 50, 10],
        "difference": ["Transmission difference", .12, .01],
    },
    "fold-ligaments": {"transmission": ["Open fold transmission", .76, .01],
                       "depth_mm": ["Fold evidence depth (mm)", 3, .1]},
}


def input_json():
    return obj(request.get_json(silent=True))


def sample_frame(library, run, photo):
    return photos.plane({**photo, "pixels": photos.decode(library.blob(photo["blob"]).read_bytes())}, run)


def measure(library, run, data):
    mode = choice(data.get("mode"), MEASUREMENTS, "reading")
    work, source_roles, other_roles = MEASUREMENTS[mode]
    source = locate(run["photos"], data.get("photo"), "The source photograph")
    source_frame = sample_frame(library, run, source)
    options = obj(data.get("options", {}))
    marked = obj(data.get("marks", {}))
    selected = {}
    other = None
    other_frame = None
    if mode in PAIRED:
        other = locate(run["photos"], data.get("other"), "The comparison photograph")
        if other["id"] == source["id"]:
            raise Invalid("Choose a different comparison photograph.")
        other_frame = sample_frame(library, run, other)
    allowed_photo_ids = {source["id"], *({other["id"]} if other else set())}

    def read_marks(assigned, capture):
        optional = OPTIONAL_ROLE_SLOTS.get(mode, set())
        for slot, purpose in assigned.items():
            if slot in optional and marked.get(slot) is None:
                continue
            mark = locate(run["marks"], marked.get(slot), MARKS[purpose])
            cross_fold = (mode in analysis.ALIGNMENT_MODE_ALIASES and slot == "fold"
                          and other is not None and mark["photo"] == other["id"])
            if mark["purpose"] != purpose or (mark["photo"] != capture["id"] and not cross_fold):
                raise Invalid(f"Select a {MARKS[purpose].lower()} mark from the specified photograph.")
            selected[slot] = mark
            if cross_fold:
                selected.setdefault("other_fold", mark)

    read_marks(source_roles, source)
    used = [source]
    if other is not None:
        read_marks(other_roles, other)
        used.append(other)

    def resolve_extra_mark(value):
        if value is None:
            return None
        if isinstance(value, str):
            mark = locate(run["marks"], value, "mark")
            if mark["photo"] not in allowed_photo_ids:
                raise Invalid("Select marks from the two photographs used by this reading.")
            return mark
        if isinstance(value, list):
            return [resolve_extra_mark(item) for item in value]
        if isinstance(value, dict):
            return value
        raise Invalid("Mark selections must be mark identifiers or mark definitions.")

    for slot, value in marked.items():
        if slot not in selected:
            selected[slot] = resolve_extra_mark(value)

    if other is not None:
        summary, rows, pixels = work(source_frame, other_frame, selected, options)
    else:
        summary, rows, pixels = work(source_frame, selected, options)
    return {
        "mode": mode, "title": READINGS[mode][0], "summary": summary, "rows": rows, "photos": used,
        "options": options, "marks": selected, "sheet_mm": [run["width_mm"], run["height_mm"]],
        "source_revision": run["revision"], "overlay": library.keep(photos.png(pixels)),
    }


def install(app, root):
    library = Library(root)
    api = Blueprint("paperbench", __name__, url_prefix="/paper")

    @app.get("/")
    def paper_desk():
        return render_template("paper.html")

    @api.get("/health")
    def health():
        return jsonify(service="paperbench", status="ready")

    @api.get("/config")
    def config():
        definitions = {}
        for mode, (title, help_text) in READINGS.items():
            _, source_roles, other_roles = MEASUREMENTS[mode]
            definitions[mode] = {"title": title, "help": help_text, "reference": source_roles, "comparison": other_roles,
                                 "paired": mode in PAIRED, "hidden": mode in HIDDEN_MODES,
                                 "options": OPTIONS.get(mode, OPTIONS["fold-alignment"] if mode in HIDDEN_MODES else {})}
        return jsonify(readings=definitions, marks=MARKS, roles=ROLES)

    @api.get("/runs")
    def runs():
        return jsonify(library.list())

    @api.post("/runs")
    def new_run():
        return jsonify(library.new_run(run_fields(input_json()))), 201

    @api.post("/demo")
    def new_demo():
        return jsonify(demo.create(library)), 201

    @api.get("/runs/<key>")
    def run(key):
        return jsonify(library.run(key))

    @api.post("/runs/<key>/photos")
    def add_photo(key):
        incoming = request.files.get("file")
        if incoming is None:
            raise Invalid("Choose a photograph to upload.")
        data = incoming.stream.read(photos.MAX_PAYLOAD + 1)
        if not data or len(data) > photos.MAX_PAYLOAD:
            raise Invalid("Use a nonempty photograph of at most 20 MiB.")
        photos.decode(data)
        title = text(request.form.get("title"), "a photograph title")
        role = choice(request.form.get("role"), ROLES, "photograph role")
        try:
            revision = int(request.form.get("revision"))
        except (TypeError, ValueError) as error:
            raise Invalid("The run revision is required.") from error

        def append(run):
            if len(run["photos"]) >= 12:
                raise Invalid("This run already holds 12 sample photographs.")
            run["photos"].append({"id": identity(), "title": title, "name": title, "role": role,
                                  "filename": photos.basename(incoming.filename), "blob": library.keep(data),
                                  "corners": None, "synthetic": False, "uploaded": stamp()})

        return jsonify(library.change(key, revision, append)), 201

    @api.put("/runs/<key>/photos/<photo_key>/calibration")
    def calibrate(key, photo_key):
        data = input_json()
        corners = photos.corners(data.get("corners"))

        def update(run):
            locate(run["photos"], photo_key, "The photograph")["corners"] = corners

        return jsonify(library.change(key, data.get("revision"), update))

    @api.get("/runs/<key>/photos/<photo_key>/image")
    def image_preview(key, photo_key):
        run = library.run(key)
        photo = locate(run["photos"], photo_key, "The photograph")
        if request.args.get("view") == "source" or not photo["corners"]:
            pixels = photos.decode(library.blob(photo["blob"]).read_bytes())
        else:
            pixels = photos.image(sample_frame(library, run, photo))
        return Response(photos.png(pixels), mimetype="image/png")

    @api.get("/runs/<key>/photos/<photo_key>/source")
    def original(key, photo_key):
        run = library.run(key)
        photo = locate(run["photos"], photo_key, "The photograph")
        return send_file(library.blob(photo["blob"]), download_name=photo["filename"], as_attachment=True)

    @api.post("/runs/<key>/marks")
    def create_mark(key):
        data = input_json()
        mark = mark_fields(data)

        def change(run):
            photo = locate(run["photos"], data.get("photo"), "The photograph")
            if not photo["corners"]:
                raise Invalid("Register the paper sheet before marking its calibrated plane.")
            if len(run["marks"]) >= 100:
                raise Invalid("This run already holds 100 marks.")
            run["marks"].append({**mark, "id": identity(), "photo": photo["id"]})

        return jsonify(library.change(key, data.get("revision"), change)), 201

    @api.delete("/runs/<key>/marks/<mark_key>")
    def delete_mark(key, mark_key):
        data = input_json()

        def change(run):
            run["marks"].remove(locate(run["marks"], mark_key, "The mark"))

        return jsonify(library.change(key, data.get("revision"), change))

    @api.post("/runs/<key>/readings")
    def create_reading(key):
        data = input_json()
        run = library.run(key)
        if run["revision"] != data.get("revision"):
            raise Conflict("The run changed. Reload it before saving.")
        result = measure(library, run, data)
        updated, reading = library.save_reading(key, data["revision"], result)
        return jsonify(run=updated, reading=reading), 201

    @api.get("/runs/<key>/readings/<reading_id>")
    def reading_detail(key, reading_id):
        return jsonify(library.reading(library.run(key), reading_id))

    @api.get("/runs/<key>/readings/<reading_id>/overlay")
    def reading_overlay(key, reading_id):
        result = library.reading(library.run(key), reading_id)
        return send_file(library.blob(result["overlay"]), mimetype="image/png")

    @api.get("/runs/<key>/readings/<reading_id>/csv")
    def reading_csv(key, reading_id):
        result = library.reading(library.run(key), reading_id)
        return send_file(BytesIO(evidence.rows_csv(result)), download_name="measured-values.csv",
                         as_attachment=True, mimetype="text/csv")

    @api.get("/runs/<key>/readings/<reading_id>/packet")
    def reading_packet(key, reading_id):
        result = library.reading(library.run(key), reading_id)
        return send_file(BytesIO(evidence.packet(library, result)), download_name="paper-reading.zip",
                         as_attachment=True, mimetype="application/zip")

    app.register_blueprint(api)
    app.extensions["paper_library"] = library
