import time
import mujoco
import numpy as np

XML_PATH = "tendon_1118_three_segments.xml"  # 네 XML 파일 경로로 수정


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
    print_interval = 0.05  # 50 ms 마다 출력
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

            t = data.time
            if t - t_last_print >= print_interval:
                t_last_print = t
                print(f"\n=== t = {t:.3f} s ===")
                print("orientation (quat) [w, x, y, z]:")
                print(" ", quat)
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
