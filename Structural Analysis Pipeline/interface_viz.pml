# ================================================================
#  EFS–SRC Interface Visualization — PyMOL Script
#  Source: AlphaFold 3  |  fold_efs_and_src_model_1.cif
#  Contacts: interface_contacts.csv  (cutoff 5.0 Å)
#
#  Usage:  pymol interface_viz.pml
#  Or open PyMOL and run:  @interface_viz.pml
# ================================================================

reinitialize
load fold_efs_and_src_model_1.cif, complex

# ── Scene setup ─────────────────────────────────────────────────
bg_color white
set antialias, 2
set ray_shadows, 0
set depth_cue, 1
set cartoon_fancy_helices, 1
set cartoon_fancy_sheets, 1
set ray_opaque_maps, 0
set label_font_id,   10
set label_size,      14
set label_color,     black
set label_outline_color, white
set label_bg_transparency, 0.2

# ── Representation ──────────────────────────────────────────────
as cartoon, complex
set cartoon_transparency, 0.10

# ── Chain colours ───────────────────────────────────────────────
color marine,    complex and chain A    # EFS  — blue
color firebrick, complex and chain B    # SRC  — red

# ── Interface residues (key contacts ≤ 2.6 Å) ──────────────────
select efs_iface, complex and chain A and resi \
    101+15+62+489+382+149+99+225+482

select src_iface, complex and chain B and resi \
    391+359+461+110+158+276+426+496+128

show sticks,  efs_iface or src_iface
util.cnc      efs_iface                  # colour-by-element, keep C marine
util.cnc      src_iface                  # colour-by-element, keep C firebrick

color marine,    efs_iface and name C*
color firebrick, src_iface and name C*

# ── Distance dashes — H-bonds (yellow) ─────────────────────────
distance HB_TYR101_ARG391,  /complex//A/101/OH,   /complex//B/391/NH2
distance HB_ARG62_LYS461,   /complex//A/62/NH2,   /complex//B/461/O
distance HB_ASP149_LEU276,  /complex//A/149/N,    /complex//B/276/O
distance HB_HIS482_SER128,  /complex//A/482/NE2,  /complex//B/128/OG

color yellow, HB_TYR101_ARG391
color yellow, HB_ARG62_LYS461
color yellow, HB_ASP149_LEU276
color yellow, HB_HIS482_SER128

# ── Distance dashes — Salt Bridges (orange) ─────────────────────
distance SB_ASP15_LYS359,   /complex//A/15/OD2,   /complex//B/359/NZ
distance SB_ASP489_ARG110,  /complex//A/489/OD2,  /complex//B/110/NH1
distance SB_ASP382_ARG158,  /complex//A/382/OD1,  /complex//B/158/NH1
distance SB_GLU99_LYS426,   /complex//A/99/OE2,   /complex//B/426/NZ
distance SB_ARG225_ASP496,  /complex//A/225/NH1,  /complex//B/496/OD1

color orange, SB_ASP15_LYS359
color orange, SB_ASP489_ARG110
color orange, SB_ASP382_ARG158
color orange, SB_GLU99_LYS426
color orange, SB_ARG225_ASP496

# ── Dash style ──────────────────────────────────────────────────
set dash_width,       3.5
set dash_gap,         0.45
set dash_length,      0.65
set dash_round_ends,  on
hide labels, HB_*
hide labels, SB_*

# ── Atom labels ─────────────────────────────────────────────────
# H-bonds
label /complex//A/101/OH,   "H-bond  1.991 Å"
label /complex//A/62/NH2,   "H-bond  2.291 Å"
label /complex//A/149/N,    "H-bond  2.383 Å"
label /complex//A/482/NE2,  "H-bond  2.525 Å"

# Salt bridges
label /complex//A/15/OD2,   "Salt Bridge  2.039 Å"
label /complex//A/489/OD2,  "Salt Bridge  2.332 Å"
label /complex//A/382/OD1,  "Salt Bridge  2.377 Å"
label /complex//A/99/OE2,   "Salt Bridge  2.443 Å"
label /complex//A/225/NH1,  "Salt Bridge  2.479 Å"

# ── Zoom to interface ────────────────────────────────────────────
zoom efs_iface or src_iface, 8
orient efs_iface or src_iface

# ── Ray-trace and save (uncomment to export) ─────────────────────
# ray 2400, 1800
# png interface_contacts.png, dpi=300, ray=1

print ""
print "  EFS (Chain A) = blue / marine"
print "  SRC (Chain B) = red  / firebrick"
print "  Yellow dashes = H-bonds  (4 contacts)"
print "  Orange dashes = Salt Bridges (5 contacts)"
print ""
print "  To export:  ray 2400,1800 / png interface_contacts.png, dpi=300, ray=1"
