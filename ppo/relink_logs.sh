# run this from ppo/logs/ if the array is already running and you forgot
# the symlinks. same naming as job_ppo_jepa.sh.
GAMES=(BeamRider Breakout DemonAttack Enduro MsPacman Pong Qbert Riverraid \
       RoadRunner Seaquest SpaceInvaders)
MODES=(frozen joint)
SEED="${SEED:-0}"
for f in ppo_jepa_*_*.{out,err}; do
    [ -e "$f" ] || continue
    ext="${f##*.}"
    t="${f%.*}"; t="${t##*_}"    # task id is the last _ chunk before the extension
    ln -sfn "$f" "jepa_${MODES[$((t / 11))]}_${GAMES[$((t % 11))]}_s${SEED}.${ext}"
done
