#!/bin/bash
set -e

# ==========================================
# 1. Configuration
# ==========================================
STL_FILE="python/frame_V2.stl"
NX=2048
NY=1024
MARGIN=0.5
HEADING=-40
U_IN=0.06         
ITERATIONS=100000 
INFLOW_DIR=1      
SAVE_INTERVAL=100  # 100.000 / 100 = 1000 frames voor een vloeiende, lange video!

VIEWS=("side_left")
BASENAME=$(basename "$STL_FILE" .stl)
TEMP_DIR="temp_anim_data"

echo "---------------------------------------------------"
echo " GCA CFD Animation Pipeline - Resolution: ${NX}x${NY}"
echo "---------------------------------------------------"

rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"
mkdir -p images
# LET OP: We wissen de obstacle masks hier expres NIET meer, 
# zodat ze hergebruikt worden in stap 2!

echo ""
echo "[1/4] Compiling CUDA source..."
make fpv_cfd_stl > /dev/null 2>&1
echo "   Done."

# ==========================================
# 2. Generating Masks
# ==========================================
echo ""
echo "[2/4] Verifying 2D obstacle masks..."
for VIEW in "${VIEWS[@]}"; do
    MASK_CSV="obstacle_mask_${BASENAME}_${VIEW}_hdg${HEADING}.csv"
    if [ -f "$MASK_CSV" ]; then
        echo "   Mask '$MASK_CSV' exists. Skipping generation."
    else
        echo "   Generating new mask for view: $VIEW"
        python3 python/stl_to_mask.py --stl "$STL_FILE" --view "$VIEW" --nx "$NX" --ny "$NY" --heading-deg "$HEADING" --margin "$MARGIN" > /dev/null
    fi
done

# ==========================================
# 3. Running CUDA Simulation
# ==========================================
echo ""
echo "[3/4] Running GPU simulation..."
for VIEW in "${VIEWS[@]}"; do
    MASK_CSV="obstacle_mask_${BASENAME}_${VIEW}_hdg${HEADING}.csv"
    # De output base file blijft .csv, de C++ code maakt intern de .bin bestanden aan
    TEMP_VEL_CSV="$TEMP_DIR/velocity_stl_${VIEW}_hdg${HEADING}.csv"

    if [ -f "$MASK_CSV" ]; then
        ./fpv_cfd_stl "$MASK_CSV" "$TEMP_VEL_CSV" "$NX" "$NY" "$U_IN" "$ITERATIONS" "$INFLOW_DIR" "$SAVE_INTERVAL"
        # Kopieer het eindresultaat (de final CSV) veilig naar de hoofdmap
        cp "$TEMP_VEL_CSV" "velocity_stl_${VIEW}_hdg${HEADING}.csv"
    else
        echo "   WARNING: Mask $MASK_CSV not found!"
    fi
done

# ==========================================
# 4. Generating Animation & Cleanup
# ==========================================
echo ""
echo "[4/4] Generating MP4 animation..."
python3 python/animate_cfd.py --temp-dir "$TEMP_DIR" --out-mp4 "images/cfd_animation_${VIEWS[0]}.mp4" --nx "$NX" --ny "$NY"

echo ""
echo "[Cleanup] Removing temporary files..."
rm -rf "$TEMP_DIR"
echo "   Temporary directory removed."

echo "---------------------------------------------------"
echo " Pipeline completed successfully."
echo "---------------------------------------------------"