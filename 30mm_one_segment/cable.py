#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, csv
import numpy as np
import mujoco as mj
import mujoco.viewer

# ===== 사용자 설정 =====
XML_FILENAME = "ex4.xml"   # XML 파일명 확인!
LOG_DT       = 0.05        # 로그 주기(초)
WRITE_CSV    = False       # True면 imu_log.csv 저장
# ======================

def main():
    # ---- 언버퍼드/라인버퍼링 출력 (터미널 즉시 반영) ----
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    # ---- XML 경로 ----
    xml_path = os.path.join(os.path.dirname(__file__), XML_FILENAME)
    print(f"[INFO] XML: {xml_path}", flush=True)
    if not os.path.exists(xml_path):
        print("[ERROR] XML 파일을 찾을 수 없습니다.", flush=True)
        return

    # ---- 모델/데이터 ----
    model = mj.MjModel.from_xml_path(xml_path)
    data  = mj.MjData(model)
    mj.mj_forward(model, data)

    # ---- 센서 인벤토리 출력 ----
    print(f"[INFO] nsensor={model.nsensor}", flush=True)
    for i in range(model.nsensor):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, i)
        dim  = model.sensor_dim[i]
        styp = model.sensor_type[i]
        print(f"  - id={i:2d} name={name} type={styp} dim={dim}", flush=True)

    # ---- 우리가 원하는 센서들의 슬라이스 준비 ----
    wanted = ["imu_acc", "imu_gyro", "imu_pos", "imu_quat"]
    sid = lambda n: mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, n)

    sensors = {}
    for name in wanted:
        i = sid(name)
        if i != -1:
            a = model.sensor_adr[i]
            d = model.sensor_dim[i]
            sensors[name] = (i, a, d)

    if not sensors:
        print("[WARN] 원하는 IMU 센서(imu_acc/imu_gyro/imu_pos/imu_quat)가 XML에 없거나 이름이 다릅니다.", flush=True)

    def grab(name: str):
        _, a, d = sensors[name]
        return data.sensordata[a:a+d].copy()

    def read_imu():
        acc  = grab("imu_acc")  if "imu_acc"  in sensors else None
        gyro = grab("imu_gyro") if "imu_gyro" in sensors else None
        pos  = grab("imu_pos")  if "imu_pos"  in sensors else None
        quat = grab("imu_quat") if "imu_quat" in sensors else None
        return acc, gyro, pos, quat

    # ---- CSV 로깅(옵션) ----
    writer = None
    csvfile = None
    if WRITE_CSV:
        csvfile = open("imu_log.csv", "w", newline="")
        writer = csv.writer(csvfile)
        writer.writerow(["t","ax","ay","az","wx","wy","wz","x","y","z","qw","qx","qy","qz"])

    # ---- 공식 뷰어 실행 ----
    with mujoco.viewer.launch(model, data) as v:
        t_last = 0.0
        print("[INFO] Viewer launched. (Space: pause, H: help)", flush=True)

        while v.is_running():
            # 물리 스텝
            mj.mj_step(model, data)

            # 주기 출력
            if (data.time - t_last) >= LOG_DT:
                if sensors:
                    acc, gyro, pos, quat = read_imu()
                    def S(x): return "None" if x is None else np.array2string(x, precision=4)
                    print(f"t={data.time:7.3f}  acc={S(acc)}  gyro={S(gyro)}  pos={S(pos)}  quat={S(quat)}", flush=True)
                    if writer is not None and all(v is not None for v in (acc, gyro, pos, quat)):
                        writer.writerow([data.time, *acc, *gyro, *pos, *quat])
                else:
                    # 센서 없어도 하트비트 찍기
                    print(f"t={data.time:7.3f}  (no sensors matched)", flush=True)
                t_last = data.time

            v.sync()

    if csvfile is not None:
        csvfile.close()
    print("[INFO] Done.", flush=True)


if __name__ == "__main__":
    main()
