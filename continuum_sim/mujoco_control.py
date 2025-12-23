import time

import mujoco
import mujoco.viewer

XML_PATH = "tendon_1125_imu_test.xml" #tendon_1112_one_segment.xml

# ---- 튜닝 가능한 파라미터 ----
TENSION_BASE = 1.0   # 모든 텐던에 기본으로 걸어줄 프리텐션
GAIN          = 2.0  # 조이스틱 값 -> 힘 스케일
JOY_STEP      = 0.2  # 방향키 한번 눌렀을 때 jx/jy 증가량
JOY_LIMIT     = 1.0  # jx, jy 클리핑 범위 (-JOY_LIMIT ~ +JOY_LIMIT)
CTRL_MIN      = -3.0 # actuator ctrlrange와 맞춰줄 것
CTRL_MAX      =  3.0


def main():
    # 모델 로드
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # 조이스틱 상태 (논리적인 2축)
    state = {
        "jx": 0.0,   # 좌우
        "jy": 0.0,   # 앞뒤
        "running": True,
    }

    # GLFW 기준 키코드 값 (MuJoCo viewer에서 그대로 들어옴)
    KEY_UP    = 265
    KEY_DOWN  = 264
    KEY_RIGHT = 262
    KEY_LEFT  = 263
    KEY_ESC   = 256

    def key_callback(keycode: int):
        """뷰어 창에서 키보드 입력이 발생할 때마다 호출됨."""
        # 앞뒤
        if keycode == KEY_DOWN: #앞뒤 반대
            state["jy"] += JOY_STEP   # 앞으로
        elif keycode == KEY_UP:
            state["jy"] -= JOY_STEP   # 뒤로

        # 좌우
        elif keycode == KEY_LEFT:#좌우 반대
            state["jx"] += JOY_STEP   # 오른쪽
        elif keycode == KEY_RIGHT:
            state["jx"] -= JOY_STEP   # 왼쪽

        # R 키: 조이스틱 리셋
        elif keycode == ord('R'):
            state["jx"] = 0.0
            state["jy"] = 0.0
            print("[INFO] Joystick reset")

        # ESC: 종료 플래그
        elif keycode == KEY_ESC:
            state["running"] = False

    # 패시브 뷰어 실행 (키 콜백 등록)
    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback
    ) as viewer:

        print("Arrow keys: move continuum robot")
        print("  ↑ / ↓ : 앞뒤 굽힘")
        print("  ← / → : 좌우 굽힘")
        print("  R     : 조이스틱 리셋")
        print("  ESC   : 종료")

        while viewer.is_running() and state["running"]:
            step_start = time.time()

            # 조이스틱 값 클리핑
            jx = max(-JOY_LIMIT, min(JOY_LIMIT, state["jx"]))
            jy = max(-JOY_LIMIT, min(JOY_LIMIT, state["jy"]))
            state["jx"], state["jy"] = jx, jy

            # ---- 조이스틱 -> 4개 텐던 힘으로 매핑 ----
            # 동(East)  = cable_red    = tendon_1
            # 서(West)  = cable_green  = tendon_2
            # 북(North) = cable_blue   = tendon_3
            # 남(South) = cable_yellow = tendon_4

            u_E = TENSION_BASE + GAIN * jx   # 오른쪽 당기기
            u_W = TENSION_BASE - GAIN * jx   # 왼쪽 당기기
            u_N = TENSION_BASE + GAIN * jy   # 앞으로(북쪽) 당기기
            u_S = TENSION_BASE - GAIN * jy   # 뒤로(남쪽) 당기기

            # actuator ctrlrange에 맞게 클리핑
            def clamp(u):
                return max(CTRL_MIN, min(CTRL_MAX, u))

            u_E = clamp(u_E)
            u_W = clamp(u_W)
            u_N = clamp(u_N)
            u_S = clamp(u_S)

            # ---- MuJoCo data.ctrl에 쓰기 ----
            data.ctrl[model.actuator('cable_red').id]    = u_E
            data.ctrl[model.actuator('cable_green').id]  = u_W
            data.ctrl[model.actuator('cable_blue').id]   = u_N
            data.ctrl[model.actuator('cable_yellow').id] = u_S

            # 물리 시뮬레이션 한 스텝
            mujoco.mj_step(model, data)

            # 뷰어와 상태 동기화
            viewer.sync()

            # (선택) 시뮬레이션 속도 맞추고 싶으면 타임스텝 맞춰서 sleep
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
