import time
import mujoco
import mujoco.viewer
import numpy as np

XML_PATH = "tendon_1125_imu_test.xml"

TENSION_BASE = 1.0
GAIN          = 2.0
JOY_STEP      = 0.2
JOY_LIMIT     = 1.0
CTRL_MIN      = -3.0
CTRL_MAX      =  3.0

def quat_to_rpy_zyx(q):
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.sign(sinp) * (np.pi / 2.0)
    else:
        pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # --- IMU sensor index 준비 ---
    id_quat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat_seg1")
    adr_quat = model.sensor_adr[id_quat]

    # actuator id도 준비 (이렇게 해도 되고, 네 코드처럼 model.actuator(...) 써도 됨)
    id_red   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cable_red")
    id_green = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cable_green")
    id_blue  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cable_blue")
    id_yellow= mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cable_yellow")

    state = {"jx": 0.0, "jy": 0.0, "running": True}

    KEY_UP    = 265
    KEY_DOWN  = 264
    KEY_RIGHT = 262
    KEY_LEFT  = 263
    KEY_ESC   = 256

    def key_callback(keycode: int):
        if keycode == KEY_DOWN:
            state["jy"] += JOY_STEP
        elif keycode == KEY_UP:
            state["jy"] -= JOY_STEP
        elif keycode == KEY_LEFT:
            state["jx"] += JOY_STEP
        elif keycode == KEY_RIGHT:
            state["jx"] -= JOY_STEP
        elif keycode == ord('R'):
            state["jx"] = 0.0
            state["jy"] = 0.0
            print("[INFO] Joystick reset")
        elif keycode == KEY_ESC:
            state["running"] = False

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback
    ) as viewer:

        print("Arrow keys: move continuum robot")
        print("  ↑ / ↓ : 앞뒤 굽힘")
        print("  ← / → : 좌우 굽힘")
        print("  R     : 조이스틱 리셋")
        print("  ESC   : 종료")

        t_last_print = 0.0
        print_interval = 0.05  # 50 ms 정도마다 IMU 출력

        while viewer.is_running() and state["running"]:
            step_start = time.time()

            jx = max(-JOY_LIMIT, min(JOY_LIMIT, state["jx"]))
            jy = max(-JOY_LIMIT, min(JOY_LIMIT, state["jy"]))
            state["jx"], state["jy"] = jx, jy

            u_E = TENSION_BASE + GAIN * jx
            u_W = TENSION_BASE - GAIN * jx
            u_N = TENSION_BASE + GAIN * jy
            u_S = TENSION_BASE - GAIN * jy

            def clamp(u):
                return max(CTRL_MIN, min(CTRL_MAX, u))

            u_E = clamp(u_E)
            u_W = clamp(u_W)
            u_N = clamp(u_N)
            u_S = clamp(u_S)

            data.ctrl[id_red]    = u_E
            data.ctrl[id_green]  = u_W
            data.ctrl[id_blue]   = u_N
            data.ctrl[id_yellow] = u_S

            mujoco.mj_step(model, data)

            # --- 여기서 IMU 읽기 ---
            sensors = data.sensordata
            quat = sensors[adr_quat: adr_quat+4]
            rpy_rad = quat_to_rpy_zyx(quat)
            rpy_deg = np.degrees(rpy_rad)

            t = data.time
            if t - t_last_print >= print_interval:
                t_last_print = t
                print(f"\n=== t = {t:.3f} s ===")
                print("qpos:", data.qpos.copy())
                print("quat:", quat)
                print("rpy(deg) [roll, pitch, yaw]:", rpy_deg)

            viewer.sync()

            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)

if __name__ == "__main__":
    main()
