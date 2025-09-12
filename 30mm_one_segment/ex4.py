#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import numpy as np
import mujoco as mj
from mujoco.glfw import glfw

# ========= 사용자 설정 파라미터 ======================================
XML_FILENAME = "ex4.xml"     # <-- 당신의 XML 파일명으로 변경
SIM_FPS      = 60              # 렌더 프레임 기준
LOG_DT       = 0.02            # 센서 로그 간격(초)
WRITE_CSV    = False           # True면 imu_log.csv에 저장
# =================================================================



def main():
    # XML 경로
    xml_path = os.path.join(os.path.dirname(__file__), XML_FILENAME)

    # 모델/데이터 로딩
    model = mj.MjModel.from_xml_path(xml_path)
    data  = mj.MjData(model)

    # 모델 이름(뷰어 제목용): name 속성 대신 mj_id2name 사용
    try:
        model_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_MODEL, 0)
    except Exception:
        model_name = None
    title = f"MuJoCo : {model_name or os.path.basename(xml_path)}"

    # GLFW 초기화 및 창 생성
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    window = glfw.create_window(1200, 900, title, None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW window creation failed")
    glfw.make_context_current(window)
    glfw.swap_interval(1)  # v-sync

    # 시각화 구성
    cam = mj.MjvCamera()
    opt = mj.MjvOption()
    mj.mjv_defaultCamera(cam)
    mj.mjv_defaultOption(opt)
    scene   = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_100.value)

    # 카메라 세팅(원하면 조정)
    cam.azimuth   = 90.0
    cam.elevation = -20.0
    cam.distance  = 2.5
    cam.lookat    = np.array([0.0, 0.0, 0.6])

    # 초기 forward
    mj.mj_forward(model, data)

    # ===== 센서 헬퍼 =====
    # XML에 아래 이름의 센서가 정의되어 있어야 함:
    #  <accelerometer site="S_last" name="imu_acc"/>
    #  <gyro          site="S_last" name="imu_gyro"/>
    #  <framepos  objtype="site" objname="S_last" name="imu_pos"/>
    #  <framequat objtype="site" objname="S_last" name="imu_quat"/>
    sensor_names = ["imu_acc", "imu_gyro", "imu_pos", "imu_quat"]

    sid = lambda n: mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, n)
    adr = model.sensor_adr
    dim = model.sensor_dim

    # 존재 확인(오타 방지)
    for name in sensor_names:
        si = sid(name)
        if si == -1:
            raise RuntimeError(f"Sensor not found: {name}  (XML <sensor> name 확인)")

    def grab(name: str):
        i = sid(name)
        a = adr[i]
        d = dim[i]
        return data.sensordata[a:a+d].copy()

    def read_imu():
        acc  = grab("imu_acc")   # [ax, ay, az]  (site 로컬 프레임, 3축 가속도)
        gyro = grab("imu_gyro")  # [wx, wy, wz]  (site 로컬 프레임, 3축 각속도)
        pos  = grab("imu_pos")   # [x, y, z]     (월드 좌표)
        quat = grab("imu_quat")  # [w, x, y, z]  (월드→프레임 회전)
        return acc, gyro, pos, quat

    # ===== CSV 로거(옵션) =====
    writer = None
    csvfile = None
    if WRITE_CSV:
        csvfile = open("imu_log.csv", "w", newline="")
        writer = csv.writer(csvfile)
        writer.writerow(["t",
                         "ax","ay","az",
                         "wx","wy","wz",
                         "x","y","z",
                         "qw","qx","qy","qz"])

    # ===== 메인 루프 =====
    sim_dt = 1.0 / SIM_FPS
    t_last_log = 0.0

    while not glfw.window_should_close(window):
        simstart = data.time

        # 물리 스텝: SIM_FPS에 맞춤
        while (data.time - simstart) < sim_dt:
            mj.mj_step(model, data)

        # 센서 읽기(주기 LOG_DT)
        if (data.time - t_last_log) >= LOG_DT:
            acc, gyro, pos, quat = read_imu()
            print(f"t={data.time:7.3f}  "
                  f"acc={np.array2string(acc, precision=4)}  "
                  f"gyro={np.array2string(gyro, precision=4)}  "
                  f"pos={np.array2string(pos, precision=4)}  "
                  f"quat={np.array2string(quat, precision=4)}")
            if writer is not None:
                writer.writerow([data.time, *acc, *gyro, *pos, *quat])
            t_last_log = data.time

        # 렌더링
        vw, vh = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, vw, vh)
        mj.mjv_updateScene(model, data, opt, None, cam,
                           mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)

        glfw.swap_buffers(window)
        glfw.poll_events()

    # 정리
    if csvfile is not None:
        csvfile.close()
    glfw.terminate()


if __name__ == "__main__":
    main()