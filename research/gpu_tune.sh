#!/bin/bash
# RX 6750 XT tuning (RDNA2). Run with sudo bash.
#   grub       enable overdrive in GRUB (reboot after)
#   apply      -35 mV offset + 2400 MHz max clock; fan back to firmware auto
#   fancurve   run a steeper fan curve on junction temp (foreground; Ctrl+C returns fan to auto)
#   install    systemd services: apply at boot + fan curve always on
#   uninstall  remove the services, fan back to auto
#   status     show clocks, offset, fan, temps
#   reset      stock voltage/clock, fan auto
#
# RDNA2 on Linux has no editable firmware fan curve (no gpu_od/fan_ctrl), so "fancurve"
# drives pwm itself: it never goes below the firmware's behaviour at high temp, ramps
# up fast, ramps down slowly, and on any exit restores pwm1_enable=2 (firmware auto).
[ -z "${BASH_VERSION:-}" ] && exec bash "$0" "$@"
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run as root: sudo bash $0 ${1:-grub|apply|fancurve|install|uninstall|status|reset}"; exit 1; }

SELF=$(readlink -f "$0")
D=/sys/class/drm/card1/device
H=$(ls -d $D/hwmon/hwmon*)

# junction temp (C) -> pwm (0-255); linear between points
CURVE_T=(50 60 70 80 88 95)
CURVE_P=(54 81 117 162 207 230)   # -10% vs first curve (user 2026-10-21)

curve_pwm() {
  local t=$1 n=${#CURVE_T[@]} i
  if (( t <= CURVE_T[0] )); then echo "${CURVE_P[0]}"; return; fi
  for (( i = 1; i < n; i++ )); do
    if (( t <= CURVE_T[i] )); then
      local t0=${CURVE_T[i-1]} t1=${CURVE_T[i]} p0=${CURVE_P[i-1]} p1=${CURVE_P[i]}
      echo $(( p0 + (p1 - p0) * (t - t0) / (t1 - t0) ))
      return
    fi
  done
  echo "${CURVE_P[n-1]}"
}

fan_auto() { echo 2 > $H/pwm1_enable; }

case "${1:-}" in
  grub)
    cp /etc/default/grub /etc/default/grub.bak-20261021
    sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"$/GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amdgpu.ppfeaturemask=0xffffffff"/' /etc/default/grub
    grep -n GRUB_CMDLINE_LINUX_DEFAULT /etc/default/grub
    update-grub
    echo "OK. Backup: /etc/default/grub.bak-20261021. Reboot, then: sudo bash $SELF apply"
    ;;
  apply)
    test -e $D/pp_od_clk_voltage || { echo "pp_od_clk_voltage missing: overdrive not enabled (reboot done?)"; exit 1; }
    echo manual > $D/power_dpm_force_performance_level
    echo "vo -35" > $D/pp_od_clk_voltage
    echo "s 1 2400" > $D/pp_od_clk_voltage
    echo c > $D/pp_od_clk_voltage
    # Memory stuck at 456 of 1124 MHz (168 GB/s) was the monitor at 200 Hz, not this script: the display
    # driver cannot switch memory clock inside vblank and locks it. At 144 Hz: 1124 MHz, 419 GB/s (2026-10-22).
    # Mask, pin, COMPUTE profile and "auto" all stayed at 456 while at 200 Hz.
    echo "0 1 2 3" > $D/pp_dpm_mclk || echo "WARN: mclk mask refused"
    echo 5 > $D/pp_power_profile_mode || echo "WARN: COMPUTE profile refused"
    echo "profile: $(grep '\*' $D/pp_power_profile_mode)"
    grep -q "200.0[0-9]\*" <(DISPLAY=${DISPLAY:-:0} xrandr --current 2>/dev/null) && echo "WARN: monitor at 200 Hz locks memory at 456 MHz"
    echo "0 1" > $D/pp_dpm_sclk
    systemctl is-active -q gpu-fan-curve.service || fan_auto   # keep the curve service in charge
    /bin/bash "$SELF" status
    ;;
  fancurve)
    trap 'fan_auto; echo "fan back to auto"' EXIT
    trap 'exit 0' INT TERM
    echo 1 > $H/pwm1_enable
    cur=$(cat $H/pwm1)
    while true; do
      t=$(( $(cat $H/temp2_input) / 1000 ))
      want=$(curve_pwm "$t")
      if (( want > cur )); then
        cur=$want                      # heat: react at once
      elif (( want < cur )); then
        cur=$(( cur - 3 > want ? cur - 3 : want ))   # cool: step down slowly
      fi
      echo "$cur" > $H/pwm1
      sleep 1
    done
    ;;
  install)
    cat > /etc/systemd/system/gpu-tune.service <<EOF
[Unit]
Description=RX 6750 XT undervolt and clock cap
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/bash $SELF apply

[Install]
WantedBy=multi-user.target
EOF
    cat > /etc/systemd/system/gpu-fan-curve.service <<EOF
[Unit]
Description=RX 6750 XT steeper fan curve (junction based)
After=gpu-tune.service

[Service]
ExecStart=/bin/bash $SELF fancurve
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable --now gpu-tune.service gpu-fan-curve.service
    systemctl --no-pager status gpu-fan-curve.service | head -5
    ;;
  uninstall)
    systemctl disable --now gpu-fan-curve.service gpu-tune.service 2>/dev/null || true
    rm -f /etc/systemd/system/gpu-fan-curve.service /etc/systemd/system/gpu-tune.service
    systemctl daemon-reload
    fan_auto
    echo "services removed, fan auto"
    ;;
  status)
    grep -A2 -E "OD_SCLK|OD_VDDGFX_OFFSET" $D/pp_od_clk_voltage || true
    echo "perf_level=$(cat $D/power_dpm_force_performance_level)"
    echo "fan mode=$(cat $H/pwm1_enable) (1=manual/curve 2=auto) pwm=$(cat $H/pwm1) rpm=$(cat $H/fan1_input)"
    echo "mclk now: $(grep '\*' $D/pp_dpm_mclk)  sclk now: $(grep '\*' $D/pp_dpm_sclk)"
    echo "edge=$(( $(cat $H/temp1_input)/1000 ))C junction=$(( $(cat $H/temp2_input)/1000 ))C power=$(( $(cat $H/power1_average)/1000000 ))W"
    ;;
  reset)
    echo r > $D/pp_od_clk_voltage
    echo c > $D/pp_od_clk_voltage
    echo auto > $D/power_dpm_force_performance_level
    fan_auto
    echo "defaults restored"
    ;;
  *)
    echo "usage: sudo bash $SELF grub|apply|fancurve|install|uninstall|status|reset"; exit 1
    ;;
esac
