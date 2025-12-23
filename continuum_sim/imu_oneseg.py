import time
import mujoco
import numpy as np

XML_PATH = "tendon_1125_imu_test.xml"  # 네 XML 파일 경로로 수정 tendon_1112_one_segment.xml

# --- 쿼터니언 -> 오일러각(roll, pitch, yaw) 변환 함수 ---
# Z-Y-X 순서 (yaw-pitch-roll), 단위: rad
def quat_to_rpy_zyx(q):
    """
    q: [w, x, y, z]
    return: [roll, pitch, yaw] (rad)
    """
    w, x, y, z = q

    # roll (x축 회전)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y축 회전)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        # asin 범위를 넘어가는 경우, 부호만 맞춰서 ±90도로 clamp
        pitch = np.sign(sinp) * (np.pi / 2.0)
    else:
        pitch = np.arcsin(sinp)

    # yaw (z축 회전)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def main():
    # 1) 모델 & 데이터 로드
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # 2) 센서 ID 얻기 (이름은 XML의 name과 정확히 일치해야 함)
    id_quat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat_seg1")
    id_gyro = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro_seg1")
    id_acc  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_acc_seg1")

    # 3) 각 센서의 시작 index (address) 얻기
    adr_quat = model.sensor_adr[id_quat]  # 길이 4
    adr_gyro = model.sensor_adr[id_gyro]  # 길이 3
    adr_acc  = model.sensor_adr[id_acc]   # 길이 3

    print("sensor_adr:")
    print("  quat adr:", adr_quat)
    print("  gyro adr:", adr_gyro)
    print("  acc  adr:", adr_acc)

    dt = model.opt.timestep
    print_interval = 0.01  # 50 ms 마다 출력
    t_last_print = 0.0

    try:
        while True:   # 무한 루프, Ctrl+C로 종료
            # 한 스텝 진행
            mujoco.mj_step(model, data)

            # sensordata에서 값 읽기
            sensors = data.sensordata

            quat = sensors[adr_quat : adr_quat + 4]      # [w, x, y, z]
            gyro = sensors[adr_gyro : adr_gyro + 3]      # [wx, wy, wz]
            acc  = sensors[adr_acc  : adr_acc  + 3]      # [ax, ay, az]

            # --- 쿼터니언 -> 각도 계산 ---
            rpy_rad = quat_to_rpy_zyx(quat)
            rpy_deg = np.degrees(rpy_rad)   # rad -> deg

            t = data.time
            if t - t_last_print >= print_interval:
                t_last_print = t
                print(f"\n=== t = {t:.3f} s ===")
                print("orientation (quat) [w, x, y, z]:")
                print(" ", quat)
                print("qpos:", data.qpos.copy())   # ← 추가

                print("orientation (rpy) [roll, pitch, yaw] (rad):")
                print(" ", rpy_rad)
                print("orientation (rpy) [roll, pitch, yaw] (deg):")
                print(" ", rpy_deg)
                print("gyro (rad/s) [wx, wy, wz]:")
                print(" ", gyro)
                print("acc (m/s^2) [ax, ay, az]:")
                print(" ", acc)

            # 너무 빨리 돌지 않게 약간 sleep
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n종료합니다.")


if __name__ == "__main__":
    main()
