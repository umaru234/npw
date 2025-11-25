# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 22:51:49 2025

@author: sgim4
"""

import subprocess
import json
import os
import glob
import pandas as pd
import numpy as np

# ==================== 기본 설정 ====================

SCREP_PATH = r"C:\Users\korrl\OneDrive\npw\screp.exe"  # 환경에 맞게 수정

# 슬롯 → 국가 코드 매핑 (현재 맵 기준)
SLOT_TO_NATION = {
    0: "gbr",
    1: "rus",
    2: "fra",
    3: "usa",
    4: "spa",
    5: "hre",
    6: "com",   # 컴퓨터
    7: "tur",
}

# 팀 구성
TEAM_ALLIES = {"gbr", "rus", "spa", "hre"}
TEAM_REVOLT = {"fra", "usa", "tur"}

NATION_TO_TEAM = {}
for n in TEAM_ALLIES:
    NATION_TO_TEAM[n] = "allies"
for n in TEAM_REVOLT:
    NATION_TO_TEAM[n] = "revolt"

# nation 리스트 (com 제외)
NATIONS = ["gbr", "rus", "fra", "usa", "spa", "hre", "tur"]

# 채팅/나가기 판정 관련
EARLY_LEAVE_SEC = 180  # 3분 이내 나간 사람은 탈주로 간주
GG_PATTERNS = ["ㅈㅈ", "ㅈ", "gg", "GG", "줴줴이", "ㅈㅈㅈ"]  # 필요하면 "gg", "GG" 등 추가

# 시간/좌표 관련
FPS = 24          # 1초당 프레임 수
TILE_SIZE = 32    # 타일 크기
TRAIT_MAX_RADIUS = TILE_SIZE * 1.5   # trait/이벤트는 이 정도까지 근처 클릭 허용

# 이동/이동공격 커맨드
MOVE_CMD_NAMES = {
    "Right Click",
    "Move",
    "AttackMove",
    "Patrol",
    "Attack1",
    "Attack2",
}

# 업그레이드 묶기
ATK_UPGRADES = {
    "Terran Infantry Weapons",
    "Protoss Ground Weapons",
    "Zerg Melee Attacks",
    "Zerg Missile Attacks",
}

DEF_UPGRADES = {
    "Terran Infantry Armor",
    "Protoss Ground Armor",
    "Zerg Carapace",
}

# --- 업그레이드 이름 상수 ---
UP_TERRAN_INFANTRY_WEAPONS = "Terran Infantry Weapons"
UP_PROTOSS_GROUND_WEAPONS  = "Protoss Ground Weapons"
UP_ZERG_MELEE_ATTACKS      = "Zerg Melee Attacks"
UP_ZERG_MISSILE_ATTACKS    = "Zerg Missile Attacks"

# 방어 업그레이드 이름 상수
UP_TERRAN_INFANTRY_ARMOR   = "Terran Infantry Armor"
UP_PROTOSS_GROUND_ARMOR    = "Protoss Ground Armor"
UP_ZERG_CARAPACE           = "Zerg Carapace"

ATK_UPGRADES = {
    UP_TERRAN_INFANTRY_WEAPONS,
    UP_PROTOSS_GROUND_WEAPONS,
    UP_ZERG_MELEE_ATTACKS,
    UP_ZERG_MISSILE_ATTACKS,
}

DEF_UPGRADES = {
    UP_TERRAN_INFANTRY_ARMOR,
    UP_PROTOSS_GROUND_ARMOR,
    UP_ZERG_CARAPACE,
}

UPGRADE_SPAM_WINDOW_SEC = 1.5  # "짧은 시간" 기준 (2초로 가정, 필요하면 1~3초로 조절)
UPGRADE_SPAM_CAP_PER_WINDOW = 1 # 한 윈도우(뭉치) 당 최대 인정 클릭 수
# ==================== 좌표 설정 (Trait / Focus / Event) ====================

# --- Trait 영역 ---
GBR_TRAIT_REGIONS = {
    0: {"x_min": 3712, "x_max": 3744, "y_min":  32, "y_max":  64},  # t0
    1: {"x_min": 3712, "x_max": 3744, "y_min": 160, "y_max": 192},  # t1
    2: {"x_min": 3648, "x_max": 3680, "y_min":  96, "y_max": 128},  # t2
}
USA_TRAIT_REGIONS = {
    0: {"x_min": 3808, "x_max": 3840, "y_min":  32, "y_max":  64},  # t0
    1: {"x_min": 3808, "x_max": 3840, "y_min": 160, "y_max": 192},  # t1
}
FRA_TRAIT_REGIONS = {
    0: {"x_min": 3968, "x_max": 4000, "y_min":  32, "y_max":  64},  # t0
    1: {"x_min": 3968, "x_max": 4000, "y_min": 160, "y_max": 192},
    2: {"x_min": 3904, "x_max": 3936, "y_min":  96, "y_max": 128},  # t2
}
RUS_TRAIT_REGIONS = {
    0: {"x_min": 4096, "x_max": 4128, "y_min":  32, "y_max":  64},  # t0
    1: {"x_min": 4096, "x_max": 4128, "y_min": 160, "y_max": 192},  # t1
}
SPA_TRAIT_REGIONS = {
    0: {"x_min": 4192, "x_max": 4224, "y_min":  32, "y_max":  64},  # t0
    1: {"x_min": 4192, "x_max": 4224, "y_min": 160, "y_max": 192},  # t1
}
HRE_TRAIT_REGIONS = {
    0: {"x_min": 4288, "x_max": 4320, "y_min":  32, "y_max":  64},  # t0
    1: {"x_min": 4288, "x_max": 4320, "y_min": 160, "y_max": 192},  # t1
}
TUR_TRAIT_REGIONS = {
    0: {"x_min": 4384, "x_max": 4416, "y_min":  32, "y_max":  64},  # t0
    1: {"x_min": 4384, "x_max": 4416, "y_min": 160, "y_max": 192},  # t1
}

TRAIT_REGIONS_BY_NATION = {
    "gbr": GBR_TRAIT_REGIONS,
    "usa": USA_TRAIT_REGIONS,
    "fra": FRA_TRAIT_REGIONS,
    "rus": RUS_TRAIT_REGIONS,
    "spa": SPA_TRAIT_REGIONS,
    "hre": HRE_TRAIT_REGIONS,
    "tur": TUR_TRAIT_REGIONS,
}

# --- Focus 영역 ---
GBR_FOCUS_REGIONS = {
    0: {"x_min": 3712, "x_max": 3744, "y_min":  32, "y_max":  64},  # F0
    1: {"x_min": 3712, "x_max": 3744, "y_min": 160, "y_max": 192},  # F1
}
FRA_FOCUS_REGIONS = {
    0: {"x_min": 3968, "x_max": 4000, "y_min":  32, "y_max":  64},  # F0
    1: {"x_min": 3968, "x_max": 4000, "y_min": 160, "y_max": 192},  # F1
}
USA_FOCUS_REGIONS = {
    0: {"x_min": 4960, "x_max": 4992, "y_min":  32, "y_max":  64},  # F0
    #0: {"x_min": 4960, "x_max": 4992, "y_min":  0, "y_max":  32},  # F0
    1: {"x_min": 4960, "x_max": 4992, "y_min":  160, "y_max":  192},  # F1
    #1: {"x_min": 4960, "x_max": 4992, "y_min":  64, "y_max":  92},  # F1
}
SPA_FOCUS_REGIONS = {
    0: {"x_min": 5024, "x_max": 5056, "y_min":  32, "y_max":  64},  # F0
    #0: {"x_min": 5024, "x_max": 5056, "y_min":  0, "y_max":  32},  # F0
    1: {"x_min": 5024, "x_max": 5056, "y_min":  160, "y_max":  192},  # F1
    #1: {"x_min": 5024, "x_max": 5056, "y_min":  64, "y_max":  92},  # F1
}
RUS_FOCUS_REGIONS = RUS_TRAIT_REGIONS
HRE_FOCUS_REGIONS = HRE_TRAIT_REGIONS
TUR_FOCUS_REGIONS = TUR_TRAIT_REGIONS

FOCUS_REGIONS_BY_NATION = {
    "gbr": GBR_FOCUS_REGIONS,
    "fra": FRA_FOCUS_REGIONS,
    "usa": USA_FOCUS_REGIONS,
    "spa": SPA_FOCUS_REGIONS,
    "rus": RUS_FOCUS_REGIONS,
    "hre": HRE_FOCUS_REGIONS,
    "tur": TUR_FOCUS_REGIONS,
}

# --- 시간 창(초) ---
TRAIT_TIME_WINDOW = (0, 40)

FOCUS_TIME_WINDOWS = {
    "usa": (0, 40),
    "spa": (0, 40),
    "gbr": (200, 230),  # 3:20 ~ 3:50
    "fra": (200, 230),
    "hre": (200, 230),
    "rus": (405, None),  # 6:45 이후
    "tur": (405, None),
}

# --- 이벤트(SWEDEN, BALKAN, DENMARK, CANADA, WARSAW) ---
SWEDEN_EVENT_REGIONS  = HRE_TRAIT_REGIONS   # HRE 선택
BALKAN_EVENT_REGIONS  = TUR_TRAIT_REGIONS   # TUR 선택
DENMARK_EVENT_REGIONS = USA_FOCUS_REGIONS   # USA 선택
CANADA_EVENT_REGIONS  = GBR_TRAIT_REGIONS   # GBR 선택
WARSAW_EVENT_REGIONS  = FRA_TRAIT_REGIONS   # FRA 선택

EVENT_TIME_WINDOWS = {
    "sweden":  (40, 70),
    "balkan":  (40, 70),
    "denmark": (90, 180),   # 1:30 ~ 3:00
    "canada":  (235, None),  # 4:00 ~ 6:00
    "warsaw":  (235, None),
}

#===영국 덴마크 상륙 여부 ====
GBR_DROP_REGION = {
    "x_min": 4864,
    "x_max": 5408,
    "y_min": 2240,
    "y_max": 2688,
}

# 0초 ~ 250초
GBR_DROP_TIME_WINDOW = (0, 250) 
GBR_ATK_TIME_WINDOW = (0, 300)  # [sec]
GBR_DEF_SPA_TIME_WINDOW = (400.0, None)

# 미국 공격 좌표 박스
GBR_ATK_USA_REGION = {
    "x_min":  928,
    "x_max": 1984,
    "y_min": 3328,
    "y_max": 4416,
}

# 오스만 공격 좌표 박스 (여러 구역 검증 가능)
GBR_ATK_TUR_REGIONS = [
    {   # 중동&레반트
        "x_min": 6336,
        "x_max": 7936,
        "y_min": 3776,
        "y_max": 5344,
    },
    {   # 북아프리카
        "x_min": 4384,
        "x_max": 5376,
        "y_min": 4704,
        "y_max": 5600,
    },
    # 필요하면 계속 추가 가능
]
IBERIA_SPAIN_REGION = {
    "x_min": 3456,
    "x_max": 4352,
    "y_min": 3648,
    "y_max": 4352,
}

# 인도 판정에 사용할 시간 창 (초)
TUR_ATK_INDIA_TIME_WINDOW = (0, 600)  # 0 ~ 600초

# === 미국 멕시코 공격 영역/시간 설정 ===
USA_ATK_MEX_REGION = {
    "x_min": 0,
    "x_max": 768,
    "y_min": 4480,
    "y_max": 5248,
}

# 0 ~ 600초 사이 공격만 본다
USA_ATK_MEX_TIME_WINDOW = (0, 600)

# === 프랑스의 영국 본토 침공(상륙) 영역 ===
FRA_INVASION_REGION = {
    "x_min": 3872,
    "x_max": 4448,
    "y_min": 2464,
    "y_max": 2944,
}

TUR_ATK_INDIA_REGION = {
    "x_min": 7936,
    "x_max": 8192,
    "y_min": 4928,
    "y_max": 5248,
}

# 시간창: 일단 0초 이후 전체를 본다 (필요하면 end_sec 추가해서 줄이면 됨)
FRA_INVASION_TIME_WINDOW = (0, None)  # (start_sec, end_sec). end_sec=None이면 끝까지

HREFRA_BATTLE_REGION = {
    "x_min": 4832,
    "x_max": 5856,
    "y_min": 3040,
    "y_max": 3616,
}

HREFRA_BATTLE_TIME_WINDOW = (0, 300)  # [sec]

# === 오스만 이집트 공격 영역/시간 설정 ===
TUR_ATK_EGYPT_REGION = {
    "x_min": 5540,
    "x_max": 6240,
    "y_min": 4704,
    "y_max": 5344,
}

# 0 ~ 600초 사이만 본다
TUR_ATK_EGYPT_TIME_WINDOW = (0, 600)

# === 프랑스 남미 공격 영역/시간 설정 ===
FRA_ATK_SAM_REGION = {
    "x_min": 1344,
    "x_max": 2976,
    "y_min": 5440,
    "y_max": 7776,
}

# 0 ~ 600초 사이만 본다
FRA_ATK_SAM_TIME_WINDOW = (0, 600)

# === 스페인 미국 본토 공격 영역/시간 설정 ===
SPA_ATK_USA_REGION = {
    "x_min": 1024,
    "x_max": 1856,
    "y_min": 3584,
    "y_max": 4448,
}

# 0 ~ 900초 사이 공격만 본다 (필요하면 시간 조정)
SPA_ATK_USA_TIME_WINDOW = (0, 600)

#=== 프랑스 포르투갈 공격 영역/시간 설정 ===
FRA_ATK_POR_REGION = {
    "x_min": 3456,
    "x_max": 3744,
    "y_min": 3648,
    "y_max": 4256,
}

# 전체 게임 시간 기준으로 본다 (필요하면 끝 시간 넣어서 줄이면 됨)
FRA_ATK_POR_TIME_WINDOW = (400, None)  # (start_sec, end_sec). end_sec=None이면 끝까지
#프본 영역
ALLIES_ATK_FRA_REGION = {
    "x_min": 3936,
    "x_max": 4640,
    "y_min": 3104,
    "y_max": 3744,
}

# ==================== 좌표/보정 헬퍼 ====================

def nearest_region_idx_by_center(x, y, regions, max_radius):
    best_idx = None
    best_d2 = None
    for idx, r in regions.items():
        cx = (r["x_min"] + r["x_max"]) / 2.0
        cy = (r["y_min"] + r["y_max"]) / 2.0
        dx = x - cx
        dy = y - cy
        d2 = dx * dx + dy * dy
        if best_d2 is None or d2 < best_d2:
            best_idx = idx
            best_d2 = d2
    if best_idx is None:
        return np.nan
    if max_radius is not None and best_d2 > max_radius * max_radius:
        return np.nan
    return best_idx


def safe_radius_for_two_regions(reg0, reg1, margin=2.0):
    cx0 = (reg0["x_min"] + reg0["x_max"]) / 2.0
    cy0 = (reg0["y_min"] + reg0["y_max"]) / 2.0
    cx1 = (reg1["x_min"] + reg1["x_max"]) / 2.0
    cy1 = (reg1["y_min"] + reg1["y_max"]) / 2.0
    d = ((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2) ** 0.5
    return max(0.0, d / 2.0 - margin)


def compute_focus_radius_by_nation():
    mapping = {}
    for nation, regions in FOCUS_REGIONS_BY_NATION.items():
        if len(regions) == 2 and 0 in regions and 1 in regions:
            radius = safe_radius_for_two_regions(regions[0], regions[1])
        else:
            radius = TILE_SIZE
        mapping[nation] = radius
    return mapping


FOCUS_RADIUS_BY_NATION = compute_focus_radius_by_nation()


def choose_region_with_snap(x, y, regions, max_radius):
    # 1) 하드 박스 체크
    for idx, r in regions.items():
        if (r["x_min"] <= x <= r["x_max"] and
            r["y_min"] <= y <= r["y_max"]):
            return idx
    # 2) 박스 밖이면 근처로 보정
    return nearest_region_idx_by_center(x, y, regions, max_radius)


# ==================== screp 연동/기본 유틸 ====================

def run_screp(rep_path, with_cmds=True):
    args = [SCREP_PATH]
    if with_cmds:
        args.append("-cmds")
    args.append(rep_path)

    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=True
    )
    return json.loads(result.stdout)


def load_replay(rep_path):
    """
    리플레이 1개를 읽어서
    (header, players_df, cmds_df, pdescs_df, chat_df, leave_df) 반환.
    """
    data = run_screp(rep_path, with_cmds=True)

    header   = data.get("Header", {}) or {}
    players  = header.get("Players", []) or []
    commands = data.get("Commands", {}) or {}
    computed = data.get("Computed", {}) or {}

    players_df = pd.DataFrame(players)
    cmds_df    = pd.DataFrame(commands.get("Cmds", []) or [])
    pdescs_df  = pd.DataFrame(computed.get("PlayerDescs", []) or [])
    chat_df    = pd.DataFrame(computed.get("ChatCmds", []) or [])
    leave_df   = pd.DataFrame(computed.get("LeaveGameCmds", []) or [])

    return header, players_df, cmds_df, pdescs_df, chat_df, leave_df


def get_eapm_col(pdescs_df):
    if pdescs_df.empty:
        return None
    candidates = [c for c in pdescs_df.columns if str(c).lower() == "eapm"]
    if not candidates:
        return None
    return candidates[0]


def merge_players_with_descriptors(players_df, pdescs_df):
    if pdescs_df.empty:
        return players_df.copy()
    return players_df.merge(
        pdescs_df,
        left_on="ID",
        right_on="PlayerID",
        how="left"
    )


# ==================== trait / focus / 이벤트 추론 ====================

def _infer_choice_for_slot(players_df,
                           cmds_df,
                           slot_id,
                           regions,
                           start_sec,
                           end_sec,
                           fps,
                           max_radius):
    # Slot → PlayerID
    try:
        pid = players_df.loc[players_df["SlotID"] == slot_id, "ID"].iloc[0]
    except IndexError:
        return np.nan

    start_frame = int(start_sec * fps)
    if end_sec is not None:
        end_frame = int(end_sec * fps)
        time_mask = (cmds_df["Frame"] >= start_frame) & (cmds_df["Frame"] <= end_frame)
    else:
        time_mask = (cmds_df["Frame"] >= start_frame)

    cmds = cmds_df[time_mask & (cmds_df["PlayerID"] == pid)].copy()
    if cmds.empty:
        return np.nan

    def get_type_name(t):
        if isinstance(t, dict) and "Name" in t:
            return t["Name"]
        return t

    cmds["type_name"] = cmds["Type"].apply(get_type_name)
    cmds = cmds[cmds["type_name"].isin(MOVE_CMD_NAMES)].copy()
    if cmds.empty:
        return np.nan

    def extract_x(pos):
        if isinstance(pos, dict) and "X" in pos:
            return pos["X"]
        return np.nan

    def extract_y(pos):
        if isinstance(pos, dict) and "Y" in pos:
            return pos["Y"]
        return np.nan

    cmds["pos_x"] = cmds["Pos"].apply(extract_x)
    cmds["pos_y"] = cmds["Pos"].apply(extract_y)
    cmds = cmds.dropna(subset=["pos_x", "pos_y"]).copy()
    if cmds.empty:
        return np.nan

    cmds = cmds.sort_values("Frame")

    last_choice = np.nan
    for _, row in cmds.iterrows():
        x = row["pos_x"]
        y = row["pos_y"]
        idx = choose_region_with_snap(x, y, regions, max_radius)
        if not np.isnan(idx):
            last_choice = idx

    return last_choice


def infer_all_traits(players_df, cmds_df):
    result = {}
    start_sec, end_sec = TRAIT_TIME_WINDOW
    for slot_id, nation in SLOT_TO_NATION.items():
        if nation not in TRAIT_REGIONS_BY_NATION:
            continue
        regions = TRAIT_REGIONS_BY_NATION[nation]
        val = _infer_choice_for_slot(
            players_df, cmds_df,
            slot_id=slot_id,
            regions=regions,
            start_sec=start_sec,
            end_sec=end_sec,
            fps=FPS,
            max_radius=TRAIT_MAX_RADIUS,
        )
        result[f"{nation}_trait"] = val
    return result


def infer_all_focuses(players_df, cmds_df):
    result = {}
    for slot_id, nation in SLOT_TO_NATION.items():
        if nation not in FOCUS_REGIONS_BY_NATION:
            continue
        regions = FOCUS_REGIONS_BY_NATION[nation]
        start_sec, end_sec = FOCUS_TIME_WINDOWS[nation]
        max_radius = FOCUS_RADIUS_BY_NATION.get(nation, TILE_SIZE)

        val = _infer_choice_for_slot(
            players_df, cmds_df,
            slot_id=slot_id,
            regions=regions,
            start_sec=start_sec,
            end_sec=end_sec,
            fps=FPS,
            max_radius=max_radius,
        )
        result[f"{nation}_focus"] = val
    return result


def infer_events(players_df, cmds_df):
    """
    모든 이벤트(sweden, balkan, denmark, canada, warsaw)를 추론.
    기본값: 2 (이벤트 선택 못함/발생 전 종료)
    실제 선택 시: 0/1/2 ... (영역 인덱스)
    """
    res = {
        "sweden":  2,
        "balkan":  2,
        "denmark": 2,
        "canada":  2,
        "warsaw":  2,
    }

    def _infer_single(event_name, nation, regions):
        if event_name not in EVENT_TIME_WINDOWS:
            return
        start_sec, end_sec = EVENT_TIME_WINDOWS[event_name]
        slot_ids = [s for s, n in SLOT_TO_NATION.items() if n == nation]
        if not slot_ids:
            return
        slot_id = slot_ids[0]

        idx = _infer_choice_for_slot(
            players_df, cmds_df,
            slot_id=slot_id,
            regions=regions,
            start_sec=start_sec,
            end_sec=end_sec,
            fps=FPS,
            max_radius=TRAIT_MAX_RADIUS,
        )
        if not (isinstance(idx, float) and np.isnan(idx)):
            try:
                res[event_name] = int(idx)
            except Exception:
                pass

    _infer_single("sweden",  "hre", SWEDEN_EVENT_REGIONS)
    _infer_single("balkan",  "tur", BALKAN_EVENT_REGIONS)
    _infer_single("denmark", "usa", DENMARK_EVENT_REGIONS)
    _infer_single("canada",  "gbr", CANADA_EVENT_REGIONS)
    _infer_single("warsaw",  "fra", WARSAW_EVENT_REGIONS)
    
    # === 이벤트 세분화: is_XXX_on / XXX_decision 추가 ===
    for ev in ["sweden", "balkan", "denmark", "canada", "warsaw"]:
        val = res.get(ev, 2)

        # 1) 이벤트 발동 여부: 0/1
        #    - 0 또는 1이면 1
        #    - 그 외(2, NaN)는 0
        if val in (0, 1):
            is_on = 1
            decision = val
        else:
            is_on = 0
            decision = np.nan  # pandas에서 읽으면 NaN

        res[f"is_{ev}_on"] = is_on
        res[f"{ev}_decision"] = decision
        
    return res

def _extract_name_from_field(v):
    if isinstance(v, dict) and "Name" in v:
        return v["Name"]
    return None
#영국 덴마크 상륙
def detect_gbr_dropship_landing(players_df, cmds_df):
    """
    0~250초 동안, 영국(PlayerID)의 'unload' 계열 명령이
    지정 박스(GBR_DROP_REGION 근처) 안에서 한 번이라도 나오면 1, 아니면 0.

    추가로 gbr_drop_cmd_count (해당 박스 안 unload 명령 횟수)를 같이 반환.
    """
    result = {
        "gbr_dropship_landing": np.nan,
        "gbr_drop_cmd_count": np.nan,
    }

    if cmds_df is None or cmds_df.empty or players_df is None or players_df.empty:
        return result

    # 1) 영국 PlayerID 찾기 (슬롯 0이 gbr이라는 전제)
    try:
        gbr_pid = players_df.loc[players_df["SlotID"] == 0, "ID"].iloc[0]
    except IndexError:
        return result

    # 2) 시간창 → 프레임
    start_sec, end_sec = GBR_DROP_TIME_WINDOW
    start_frame = int(start_sec * FPS)
    end_frame   = int(end_sec * FPS)

    cmds = cmds_df.copy()

    # 3) Order.Name / Type.Name 통합해서 cmd_name 만들기
    if "Order" in cmds.columns:
        cmds["order_name"] = cmds["Order"].apply(_extract_name_from_field)
    else:
        cmds["order_name"] = None

    if "Type" in cmds.columns:
        cmds["type_name"] = cmds["Type"].apply(_extract_name_from_field)
    else:
        cmds["type_name"] = None

    # Order.Name 우선, 없으면 Type.Name
    cmds["cmd_name"] = cmds["order_name"].fillna(cmds["type_name"]).astype(str).str.lower()

    # 4) 영국 + 시간창 + 'unload'가 들어가는 명령
    mask_player = cmds["PlayerID"] == gbr_pid
    mask_time   = (cmds["Frame"] >= start_frame) & (cmds["Frame"] <= end_frame)
    mask_unload = cmds["cmd_name"].str.contains("unload")

    candidate = cmds[mask_player & mask_time & mask_unload].copy()
    if candidate.empty:
        result["gbr_dropship_landing"] = 0
        result["gbr_drop_cmd_count"] = 0
        return result

    # 5) 좌표 추출
    def extract_x(pos):
        if isinstance(pos, dict) and "X" in pos:
            return pos["X"]
        return np.nan

    def extract_y(pos):
        if isinstance(pos, dict) and "Y" in pos:
            return pos["Y"]
        return np.nan

    candidate["pos_x"] = candidate["Pos"].apply(extract_x)
    candidate["pos_y"] = candidate["Pos"].apply(extract_y)
    candidate = candidate.dropna(subset=["pos_x", "pos_y"])
    if candidate.empty:
        result["gbr_dropship_landing"] = 0
        result["gbr_drop_cmd_count"] = 0
        return result

    # 6) 좌표 박스(+ margin) 안에 들어온 unload 명령 개수 세기
    MARGIN = 64  # 필요하면 조정

    x_min = GBR_DROP_REGION["x_min"] - MARGIN
    x_max = GBR_DROP_REGION["x_max"] + MARGIN
    y_min = GBR_DROP_REGION["y_min"] - MARGIN
    y_max = GBR_DROP_REGION["y_max"] + MARGIN

    in_box = (
        (candidate["pos_x"] >= x_min) & (candidate["pos_x"] <= x_max) &
        (candidate["pos_y"] >= y_min) & (candidate["pos_y"] <= y_max)
    )

    count_in_box = int(in_box.sum())
    result["gbr_drop_cmd_count"] = count_in_box
    result["gbr_dropship_landing"] = 1 if count_in_box > 0 else 0

    return result


####영국 첫턴 공격 위치####
def detect_gbr_first_turn_attack(players_df, cmds_df):
    """
    0~280초 동안, 영국(PlayerID)의 attack 계열 명령 중
    미국 / 오스만 좌표 박스 안에서 발생한 횟수를 세어서

      - gbratkusa : 미국 박스 안에서 15회 이상이면 1, 아니면 0
      - gbratktur : 오스만 박스 안에서 15회 이상이면 1, 아니면 0

    디버깅용으로 각 카운트도 함께 반환한다.
    """
    MOVE_CMD_NAMES_LOWER = {name.lower() for name in MOVE_CMD_NAMES}
    result = {
        "gbratkusa": np.nan,
        "gbratktur": np.nan,
        "gbratkusa_count": np.nan,
        "gbratktur_count": np.nan,
    }

    if cmds_df is None or cmds_df.empty or players_df is None or players_df.empty:
        return result

    # 1) 영국 PlayerID 찾기 (슬롯 0 = gbr 전제)
    try:
        gbr_pid = players_df.loc[players_df["SlotID"] == 0, "ID"].iloc[0]
    except IndexError:
        return result

    # 2) 시간창 → 프레임
    start_sec, end_sec = GBR_ATK_TIME_WINDOW
    start_frame = int(start_sec * FPS)
    end_frame   = int(end_sec * FPS)

    cmds = cmds_df.copy()

    # 3) Order.Name / Type.Name 통합해서 cmd_name 만들기
    if "Order" in cmds.columns:
        cmds["order_name"] = cmds["Order"].apply(_extract_name_from_field)
    else:
        cmds["order_name"] = None

    if "Type" in cmds.columns:
        cmds["type_name"] = cmds["Type"].apply(_extract_name_from_field)
    else:
        cmds["type_name"] = None

    # Order 우선, 없으면 Type 사용
    cmds["cmd_name"] = cmds["order_name"].fillna(cmds["type_name"]).astype(str).str.lower()

        # 4) 영국 + 시간창 + 'attack' 또는 이동 계열 명령만 필터
    mask_player = cmds["PlayerID"] == gbr_pid
    mask_time   = (cmds["Frame"] >= start_frame) & (cmds["Frame"] <= end_frame)

    # 공격 계열: 이름에 'attack' 포함 (attack, attack1, attackmove 등)
    mask_attack = cmds["cmd_name"].str.contains("attack")

    # 이동 계열: Right Click, Move, Patrol 등
    MOVE_CMD_NAMES_LOWER = {"right click", "move", "patrol", "attackmove"}
    mask_move = cmds["cmd_name"].isin(MOVE_CMD_NAMES_LOWER)

    # 공격 또는 이동
    mask_relevant = mask_attack | mask_move

    candidate = cmds[mask_player & mask_time & mask_relevant].copy()
    if candidate.empty:
        result["gbratkusa"] = 0
        result["gbratktur"] = 0
        result["gbratkusa_count"] = 0
        result["gbratktur_count"] = 0
        return result


    # 5) 좌표 추출
    def extract_x(pos):
        if isinstance(pos, dict) and "X" in pos:
            return pos["X"]
        return np.nan

    def extract_y(pos):
        if isinstance(pos, dict) and "Y" in pos:
            return pos["Y"]
        return np.nan

    candidate["pos_x"] = candidate["Pos"].apply(extract_x)
    candidate["pos_y"] = candidate["Pos"].apply(extract_y)
    candidate = candidate.dropna(subset=["pos_x", "pos_y"])
    if candidate.empty:
        result["gbratkusa"] = 0
        result["gbratktur"] = 0
        result["gbratkusa_count"] = 0
        result["gbratktur_count"] = 0
        return result

    # 6) 미국 박스 안 공격 명령 수
    ux_min = GBR_ATK_USA_REGION["x_min"]
    ux_max = GBR_ATK_USA_REGION["x_max"]
    uy_min = GBR_ATK_USA_REGION["y_min"]
    uy_max = GBR_ATK_USA_REGION["y_max"]

    in_usa = (
        (candidate["pos_x"] >= ux_min) & (candidate["pos_x"] <= ux_max) &
        (candidate["pos_y"] >= uy_min) & (candidate["pos_y"] <= uy_max)
    )
    usa_count = int(in_usa.sum())

    # 7) 오스만 박스: 여러 구역(GBR_ATK_TUR_REGIONS)을 OR로 합치기
    tur_masks = []
    for reg in GBR_ATK_TUR_REGIONS:
        tx_min = reg["x_min"]
        tx_max = reg["x_max"]
        ty_min = reg["y_min"]
        ty_max = reg["y_max"]

        mask = (
            (candidate["pos_x"] >= tx_min) & (candidate["pos_x"] <= tx_max) &
            (candidate["pos_y"] >= ty_min) & (candidate["pos_y"] <= ty_max)
        )
        tur_masks.append(mask)

    if tur_masks:
        in_tur = tur_masks[0]
        for m in tur_masks[1:]:
            in_tur |= m
    else:
        # 박스가 하나도 정의 안 돼 있으면 전부 False
        in_tur = (candidate["pos_x"] * 0).astype(bool)

    tur_count = int(in_tur.sum())

    # 카운트 기록
    result["gbratkusa_count"] = usa_count
    result["gbratktur_count"] = tur_count

    # 8) 임계값 이상이면 1, 아니면 0
    THRESH = 20  # 주석이랑 맞추려면 25로 바꿔도 됨
    result["gbratkusa"] = 1 if usa_count >= THRESH else 0
    result["gbratktur"] = 1 if tur_count >= THRESH else 0

    return result
#프랑스 영본드랍
def detect_fra_invasion(players_df, cmds_df):
    """
    프랑스가 영국 본토(정해진 박스)에 상륙했는지 감지하는 함수.

    - 대상: 프랑스(PlayerID, 슬롯 2)
    - 명령: Order/Type 이름에 'moveunload'가 들어가는 명령만 사용
    - 좌표: FRA_INVASION_REGION 안에서 나온 MoveUnload 명령의 개수
    - 시간: FRA_INVASION_TIME_WINDOW (기본 0초~게임 끝)

    반환:
      {
        "fra_invasion": 0/1,   # 한 번이라도 상륙했으면 1, 아니면 0
        "fra_invasion_cmd_count": 상륙 MoveUnload 명령 횟수
      }
    """
    result = {
        "fra_invasion": np.nan,
        "fra_invasion_cmd_count": np.nan,
    }

    if cmds_df is None or cmds_df.empty or players_df is None or players_df.empty:
        return result

    # 1) 프랑스 PlayerID 찾기 (슬롯 2 = fra 전제)
    try:
        fra_pid = players_df.loc[players_df["SlotID"] == 2, "ID"].iloc[0]
    except IndexError:
        # 프랑스가 없는 경기
        result["fra_invasion"] = 0
        result["fra_invasion_cmd_count"] = 0
        return result

    # 2) 시간창 → 프레임
    start_sec, end_sec = FRA_INVASION_TIME_WINDOW
    start_frame = int(start_sec * FPS)
    if end_sec is not None:
        end_frame = int(end_sec * FPS)
        mask_time = (cmds_df["Frame"] >= start_frame) & (cmds_df["Frame"] <= end_frame)
    else:
        mask_time = (cmds_df["Frame"] >= start_frame)

    cmds = cmds_df.copy()

    # 3) Order.Name / Type.Name 통합해서 cmd_name 만들기
    if "Order" in cmds.columns:
        cmds["order_name"] = cmds["Order"].apply(_extract_name_from_field)
    else:
        cmds["order_name"] = None

    if "Type" in cmds.columns:
        cmds["type_name"] = cmds["Type"].apply(_extract_name_from_field)
    else:
        cmds["type_name"] = None

    # Order 우선, 없으면 Type 사용
    cmds["cmd_name"] = cmds["order_name"].fillna(cmds["type_name"]).astype(str).str.lower()

    # 4) 프랑스 + 시간창 + 'moveunload'가 들어가는 명령만 필터
    mask_player = cmds["PlayerID"] == fra_pid
    mask_moveunload = cmds["cmd_name"].str.contains("moveunload")

    candidate = cmds[mask_player & mask_time & mask_moveunload].copy()
    if candidate.empty:
        result["fra_invasion"] = 0
        result["fra_invasion_cmd_count"] = 0
        return result

    # 5) 좌표 추출
    def extract_x(pos):
        if isinstance(pos, dict) and "X" in pos:
            return pos["X"]
        return np.nan

    def extract_y(pos):
        if isinstance(pos, dict) and "Y" in pos:
            return pos["Y"]
        return np.nan

    candidate["pos_x"] = candidate["Pos"].apply(extract_x)
    candidate["pos_y"] = candidate["Pos"].apply(extract_y)
    candidate = candidate.dropna(subset=["pos_x", "pos_y"])
    if candidate.empty:
        result["fra_invasion"] = 0
        result["fra_invasion_cmd_count"] = 0
        return result

    # 6) 영국 본토 박스 안에 들어온 MoveUnload 명령 개수 세기
    MARGIN = 64  # 덴마크 상륙과 동일하게 여유 박스 (원하면 조절)

    x_min = FRA_INVASION_REGION["x_min"] - MARGIN
    x_max = FRA_INVASION_REGION["x_max"] + MARGIN
    y_min = FRA_INVASION_REGION["y_min"] - MARGIN
    y_max = FRA_INVASION_REGION["y_max"] + MARGIN

    in_box = (
        (candidate["pos_x"] >= x_min) & (candidate["pos_x"] <= x_max) &
        (candidate["pos_y"] >= y_min) & (candidate["pos_y"] <= y_max)
    )

    count_in_box = int(in_box.sum())
    result["fra_invasion_cmd_count"] = count_in_box
    result["fra_invasion"] = 1 if count_in_box > 0 else 0

    return result

# 프랑스 포르투갈 공격/침투 시도 여부
def fraatkpor(players_df, cmds_df):
    """
    프랑스의 포르투갈 지역 공격/이동 시도 여부를 계산.

    - 포르투갈 좌표:
        x: 3456 ~ 3712
        y: 3648 ~ 4256

    - 시간: FRA_ATK_POR_TIME_WINDOW 사용
        기본값 (0, None) → 게임 전체 시간

    - 포함하는 명령:
        - 공격 계열: Attack, AttackMove, Attack1, Attack2 (이름에 'attack' 포함)
        - 이동 계열: Right Click, Move, Patrol 등 (MOVE_CMD_NAMES에 들어있는 것)
          → 둘을 모두 합쳐서 "포르투갈 지역에 들어간 시도"로 본다.

    반환 dict:
        {
            "fraatkpor": 0 또는 1  # 한 번이라도 위 명령이 있으면 1, 아니면 0
        }
    """
    result = {
        "fraatkpor": 0,
    }

    if cmds_df is None or cmds_df.empty or players_df is None or players_df.empty:
        return result

    # 1) 프랑스 PlayerID 찾기 (슬롯 2 = fra 전제)
    try:
        fra_pid = players_df.loc[players_df["SlotID"] == 2, "ID"].iloc[0]
    except IndexError:
        return result

    cmds = cmds_df.copy()

    # Frame 없으면 시간 필터 불가
    if "Frame" not in cmds.columns:
        return result

    # 2) 시간창 필터링
    start_sec, end_sec = FRA_ATK_POR_TIME_WINDOW
    start_frame = int(start_sec * FPS)

    if end_sec is not None:
        end_frame = int(end_sec * FPS)
        time_mask = (cmds["Frame"] >= start_frame) & (cmds["Frame"] <= end_frame)
    else:
        time_mask = (cmds["Frame"] >= start_frame)

    cmds = cmds[time_mask].copy()
    if cmds.empty:
        return result

    # 3) Order.Name / Type.Name 통합 → cmd_name / cmd_name_l
    if "Order" in cmds.columns:
        cmds["order_name"] = cmds["Order"].apply(_extract_name_from_field)
    else:
        cmds["order_name"] = None

    if "Type" in cmds.columns:
        cmds["type_name"] = cmds["Type"].apply(_extract_name_from_field)
    else:
        cmds["type_name"] = None

    cmds["cmd_name"] = cmds["order_name"].fillna(cmds["type_name"]).astype(str)
    cmds["cmd_name_l"] = cmds["cmd_name"].str.lower()

    # 4) 프랑스 명령만
    fra_cmds = cmds[cmds["PlayerID"] == fra_pid].copy()
    if fra_cmds.empty:
        return result

    # 5) 좌표 추출
    def extract_x(pos):
        if isinstance(pos, dict) and "X" in pos:
            return pos["X"]
        return np.nan

    def extract_y(pos):
        if isinstance(pos, dict) and "Y" in pos:
            return pos["Y"]
        return np.nan

    if "Pos" in fra_cmds.columns:
        fra_cmds["pos_x"] = fra_cmds["Pos"].apply(extract_x)
        fra_cmds["pos_y"] = fra_cmds["Pos"].apply(extract_y)
    else:
        fra_cmds["pos_x"] = np.nan
        fra_cmds["pos_y"] = np.nan

    fra_cmds = fra_cmds.dropna(subset=["pos_x", "pos_y"])
    if fra_cmds.empty:
        return result

    # 6) 포르투갈 영역 안 명령만 필터
    x_min = FRA_ATK_POR_REGION["x_min"]
    x_max = FRA_ATK_POR_REGION["x_max"]
    y_min = FRA_ATK_POR_REGION["y_min"]
    y_max = FRA_ATK_POR_REGION["y_max"]

    in_por = (
        (fra_cmds["pos_x"] >= x_min) & (fra_cmds["pos_x"] <= x_max) &
        (fra_cmds["pos_y"] >= y_min) & (fra_cmds["pos_y"] <= y_max)
    )

    por_cmds = fra_cmds[in_por].copy()
    if por_cmds.empty:
        return result

    # 7) 이동+공격 계열 통합 필터
    move_names_lower = {name.lower() for name in MOVE_CMD_NAMES}

    def is_move_or_attack(name_l: str) -> bool:
        if not isinstance(name_l, str):
            return False
        if "attack" in name_l:  # attack, attackmove, attack1, attack2 등
            return True
        return name_l in move_names_lower  # right click, move, patrol 등

    por_cmds["is_act"] = por_cmds["cmd_name_l"].apply(is_move_or_attack)
    act_cnt = int(por_cmds["is_act"].sum())

    if act_cnt > 0:
        result["fraatkpor"] = 1

    return result
#영국 스페인본토 방어 시도
def gbrdefspa(players_df, cmds_df, fps=FPS):
    """
    영국이 스페인 전역에서 적극적으로 활동했는지 여부를 계산.

    - 영역: IBERIA_SPAIN_REGION
        x: 3456 ~ 4352
        y: 3648 ~ 4352

    - 시간: GBR_DEF_SPA_TIME_WINDOW
        기본값 (400초, None) → 400초 이후 ~ 게임 끝까지

    - 포함 명령:
        - 공격 계열: 이름에 'attack' 이 들어가는 명령 (Attack, AttackMove, Attack1, Attack2 등)
        - 이동 계열: MOVE_CMD_NAMES 에 들어 있는 모든 명령
          (Right Click, Move, Patrol 등)

    - 판정:
        - 위 조건을 만족하는 명령이 20회 이상: gbrdefspa = 1
        - 그 외: gbrdefspa = 0

    반환 dict:
        {
            "gbrdefspa": 0 또는 1
        }
    """
    result = {
        "gbrdefspa": 0,
    }

    if players_df is None or players_df.empty or cmds_df is None or cmds_df.empty:
        return result

    # --- 영국 PlayerID 찾기 (SLOT_TO_NATION 활용) ---
    slot_to_pid = {}
    for _, row in players_df.iterrows():
        slot = row.get("SlotID")
        pid  = row.get("ID")
        if slot in SLOT_TO_NATION and pid is not None:
            nation = SLOT_TO_NATION[slot]
            slot_to_pid[nation] = pid

    gbr_pid = slot_to_pid.get("gbr")
    if gbr_pid is None:
        # 영국이 없는 판이면 0 유지
        return result

    cmds = cmds_df.copy()

    # Frame / Pos 없으면 판정 불가
    if "Frame" not in cmds.columns or "Pos" not in cmds.columns:
        return result

    # --- 시간창 필터링 (400초 이후 ~ 끝까지) ---
    start_sec, end_sec = GBR_DEF_SPA_TIME_WINDOW
    start_frame = int(start_sec * fps)

    if end_sec is not None:
        end_frame = int(end_sec * fps)
        time_mask = (cmds["Frame"] >= start_frame) & (cmds["Frame"] <= end_frame)
    else:
        time_mask = (cmds["Frame"] >= start_frame)

    cmds = cmds[time_mask].copy()
    if cmds.empty:
        return result

    # --- Order.Name / Type.Name 통합 → cmd_name / cmd_name_l ---
    if "Order" in cmds.columns:
        cmds["order_name"] = cmds["Order"].apply(_extract_name_from_field)
    else:
        cmds["order_name"] = None

    if "Type" in cmds.columns:
        cmds["type_name"] = cmds["Type"].apply(_extract_name_from_field)
    else:
        cmds["type_name"] = None

    cmds["cmd_name"]   = cmds["order_name"].fillna(cmds["type_name"]).astype(str)
    cmds["cmd_name_l"] = cmds["cmd_name"].str.lower()

    # --- 영국 명령만 ---
    gbr_cmds = cmds[cmds["PlayerID"] == gbr_pid].copy()
    if gbr_cmds.empty:
        return result

    # --- 좌표 추출 ---
    def extract_x(pos):
        if isinstance(pos, dict) and "X" in pos:
            return pos["X"]
        return np.nan

    def extract_y(pos):
        if isinstance(pos, dict) and "Y" in pos:
            return pos["Y"]
        return np.nan

    gbr_cmds["pos_x"] = gbr_cmds["Pos"].apply(extract_x)
    gbr_cmds["pos_y"] = gbr_cmds["Pos"].apply(extract_y)
    gbr_cmds = gbr_cmds.dropna(subset=["pos_x", "pos_y"])
    if gbr_cmds.empty:
        return result

    # --- 스페인 전역 영역 필터 ---
    x_min = IBERIA_SPAIN_REGION["x_min"]
    x_max = IBERIA_SPAIN_REGION["x_max"]
    y_min = IBERIA_SPAIN_REGION["y_min"]
    y_max = IBERIA_SPAIN_REGION["y_max"]

    in_spain = (
        (gbr_cmds["pos_x"] >= x_min) & (gbr_cmds["pos_x"] <= x_max) &
        (gbr_cmds["pos_y"] >= y_min) & (gbr_cmds["pos_y"] <= y_max)
    )

    spa_cmds = gbr_cmds[in_spain].copy()
    if spa_cmds.empty:
        return result

    # --- 이동+공격 계열 통합 필터 ---
    move_names_lower = {name.lower() for name in MOVE_CMD_NAMES}

    def is_move_or_attack(name_l: str) -> bool:
        if not isinstance(name_l, str):
            return False
        if "attack" in name_l:  # attack, attackmove, attack1, attack2 등
            return True
        return name_l in move_names_lower  # right click, move, patrol 등

    spa_cmds["is_act"] = spa_cmds["cmd_name_l"].apply(is_move_or_attack)
    act_cnt = int(spa_cmds["is_act"].sum())

    if act_cnt >= 20:
        result["gbrdefspa"] = 1

    return result

#프랑스 유동밀기
def infer_HREFRAbattledur(players_df, cmds_df, fps=FPS):
    """
    오스트리아 전역(x: 4832~5856, y: 3040~3616, 0~300초)에서
    FRA와 HRE의 '이동/공격 명령'이 공존하는 교전 시간(초)을 추론한다.

    - 시작 시점:
      두 국가(FRA, HRE) 모두 해당 지역 내에서 처음으로 명령을 내린 뒤,
      두 번째 국가가 처음 명령을 내린 시점 (즉, 둘 다 등장한 순간).

    - 종료 시점:
      교전이 시작된 이후, 어느 한 국가의 명령이 나온 시점 t0에서
      20초 이내에 반대 국가의 명령이 해당 지역에 다시 나타나지 않으면
      t0 + 20초를 종료 시점으로 본다.
      (단, 전체 시간 창은 0~300초로 제한)

    반환:
      float (초 단위 교전 시간). FRA/HRE 둘 중 하나라도 참여하지 않으면 0.0
    """
    result = {
        "HREFRA_battledur": 0.0,
        "HREFRA_firstbattle": np.nan,
    }

    if players_df is None or players_df.empty or cmds_df is None or cmds_df.empty:
        return result

    # 1) FRA / HRE PlayerID 찾기
    try:
        fra_slot = [s for s, n in SLOT_TO_NATION.items() if n == "fra"][0]
        hre_slot = [s for s, n in SLOT_TO_NATION.items() if n == "hre"][0]
    except IndexError:
        # 맵 구조가 다르거나 슬롯 매핑이 깨진 경우
        return result

    try:
        fra_pid = players_df.loc[players_df["SlotID"] == fra_slot, "ID"].iloc[0]
        hre_pid = players_df.loc[players_df["SlotID"] == hre_slot, "ID"].iloc[0]
    except IndexError:
        # FRA/HRE 플레이어가 없으면 교전 없음
        return result

    cmds = cmds_df.copy()

    # 2) 명령 이름 통합 (Order.Name / Type.Name)
    if "Order" in cmds.columns:
        cmds["order_name"] = cmds["Order"].apply(_extract_name_from_field)
    else:
        cmds["order_name"] = None

    if "Type" in cmds.columns:
        cmds["type_name"] = cmds["Type"].apply(_extract_name_from_field)
    else:
        cmds["type_name"] = None

    cmds["cmd_name"] = cmds["order_name"].fillna(cmds["type_name"]).astype(str).str.lower()

    # 3) 시간 창 필터 (0 ~ 300초)
    start_sec, end_sec = HREFRA_BATTLE_TIME_WINDOW
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    time_mask = (cmds["Frame"] >= start_frame) & (cmds["Frame"] <= end_frame)

    # 4) 좌표 추출 및 오스트리아 전역 박스 필터
    def extract_x(pos):
        if isinstance(pos, dict) and "X" in pos:
            return pos["X"]
        return np.nan

    def extract_y(pos):
        if isinstance(pos, dict) and "Y" in pos:
            return pos["Y"]
        return np.nan

    cmds["pos_x"] = cmds["Pos"].apply(extract_x)
    cmds["pos_y"] = cmds["Pos"].apply(extract_y)

    x_min = HREFRA_BATTLE_REGION["x_min"]
    x_max = HREFRA_BATTLE_REGION["x_max"]
    y_min = HREFRA_BATTLE_REGION["y_min"]
    y_max = HREFRA_BATTLE_REGION["y_max"]

    box_mask = (
        (cmds["pos_x"] >= x_min) & (cmds["pos_x"] <= x_max) &
        (cmds["pos_y"] >= y_min) & (cmds["pos_y"] <= y_max)
    )

    # 5) 이동/공격 명령 필터
    #   - 이동: right click / move / attackmove / patrol 등
    #   - 공격: 'attack' 문자열을 포함하는 모든 명령 (attack, attack1, attack2...)
    move_names = {"right click", "move", "attackmove", "patrol"}
    is_move = cmds["cmd_name"].isin(move_names)
    is_attack = cmds["cmd_name"].str.contains("attack")

    act_mask = is_move | is_attack

    # FRA / HRE, 시간/좌표/행동 조건 모두 만족하는 명령만 남김
    sub = cmds[time_mask & box_mask & act_mask].copy()
    if sub.empty:
        return result

    fra_cmds = sub[sub["PlayerID"] == fra_pid].copy()
    hre_cmds = sub[sub["PlayerID"] == hre_pid].copy()

    if fra_cmds.empty or hre_cmds.empty:
        # 둘 중 한 쪽이라도 해당 지역에서 활동이 없으면 교전 시간 0
        return result

    # 6) Frame → 초로 변환, 시간 리스트 추출
    fra_times = np.sort(fra_cmds["Frame"].values.astype(float) / fps)
    hre_times = np.sort(hre_cmds["Frame"].values.astype(float) / fps)

    # 7) FRA/HRE 이벤트 통합 타임라인 생성
    events = []
    for t in fra_times:
        events.append((t, "fra"))
    for t in hre_times:
        events.append((t, "hre"))

    events.sort(key=lambda x: x[0])  # 시간 순 정렬

    # 8) 교전 시작 시점: 두 국가가 모두 등장한 후, 두 번째 국가가 처음 명령 내린 시점
    fra_seen = False
    hre_seen = False
    t_start = None
    start_idx = None

    for i, (t, nation) in enumerate(events):
        if nation == "fra":
            fra_seen = True
        else:
            hre_seen = True

        if t_start is None and fra_seen and hre_seen:
            t_start = t
            start_idx = i
            break

    if t_start is None:
        # 둘이 같은 지역에서 아예 겹치지 않은 경우
        return result

    # 9) 교전 종료 시점 계산
    #    - 교전 시작 이후, 어느 한 국가의 명령 시점 t0에서
    #      20초 이내에 반대 국가 명령이 다시 안 나오면 t0 + 20을 종료 시점으로 본다.
    MAX_GAP = 25.0  # sec
    t_end = None

    for i in range(start_idx, len(events)):
        t_i, nation_i = events[i]
        other = "fra" if nation_i == "hre" else "hre"

        # t_i 이후, 다른 국가의 다음 명령 시점을 찾는다.
        next_other_time = None
        for j in range(i + 1, len(events)):
            t_j, nation_j = events[j]
            if nation_j == other:
                next_other_time = t_j
                break

        # 반대편 명령이 없거나, 너무 늦게(>20초 후) 나오면 여기서 교전 종료
        if next_other_time is None or (next_other_time - t_i) > MAX_GAP:
            t_end = min(t_i + MAX_GAP, float(end_sec))
            break

    if t_end is None:
        # 끝까지 서로 어느 정도 응수하며 싸운 경우:
        # 단순히 마지막 이벤트 시점까지만 교전으로 본다 (0~300초 제한)
        last_time = events[-1][0]
        t_end = min(last_time, float(end_sec))

    battledur = max(0.0, t_end - t_start)
    result["HREFRA_battledur"] = battledur
    result["HREFRA_firstbattle"] = float(t_start)
    return result

# === 프랑스 러시아 공격 영역 설정 ===
# 범위1: x: 6176~6816, y: 1856~2592
FRA_ATK_RUS_REGION_1 = {
    "x_min": 6176,
    "x_max": 6816,
    "y_min": 1856,
    "y_max": 2592,
}

# 범위2: x: 6496~6976, y: 2592~3200
FRA_ATK_RUS_REGION_2 = {
    "x_min": 6496,
    "x_max": 6976,
    "y_min": 2592,
    "y_max": 3200,
}

#프랑스 러시아 공격
def fraatkrus(players_df, cmds_df):
    """
    프랑스의 러시아 방향 공격/진입 여부 및 관련 명령 수를 계산.

    - 러시아 방향 좌표:
        범위1: x: 6176 ~ 6816, y: 1856 ~ 2592
        범위2: x: 6496 ~ 6976, y: 2592 ~ 3200

    - 시간 제한은 두지 않고, 전체 게임 시간 동안의 명령을 봄.

    - 명령 종류:
        Pos(X,Y)를 가지는 모든 명령을 "해당 지역에서 내려진 명령"으로 봄
        (Attack, AttackMove, Right Click, Move, 건물 배치 등)

    판정:
        - 두 영역 중 어느 한 쪽이라도 프랑스 명령이 20회 이상이면 fraatkrus = 1
        - 그 명령 수는 fraatkrus_cmd_count 에 저장

    반환 dict:
        {
            "fraatkrus": 0 또는 1,
            "fraatkrus_cmd_count": 해당 영역 내 명령 수 (int)
        }
    """
    result = {
        "fraatkrus": 0,
        "fraatkrus_cmd_count": 0,
    }

    if cmds_df is None or cmds_df.empty or players_df is None or players_df.empty:
        return result

    # 1) 프랑스 PlayerID 찾기 (슬롯 2 = fra 전제)
    try:
        fra_pid = players_df.loc[players_df["SlotID"] == 2, "ID"].iloc[0]
    except IndexError:
        # 프랑스가 아예 없으면 그대로 0 반환
        return result

    cmds = cmds_df.copy()

    # 좌표 없는 로그면 계산 불가
    if "Pos" not in cmds.columns:
        return result

    # 2) 프랑스 명령만
    fra_cmds = cmds[cmds["PlayerID"] == fra_pid].copy()
    if fra_cmds.empty:
        return result

    # 3) 좌표 추출
    def extract_x(pos):
        if isinstance(pos, dict) and "X" in pos:
            return pos["X"]
        return np.nan

    def extract_y(pos):
        if isinstance(pos, dict) and "Y" in pos:
            return pos["Y"]
        return np.nan

    fra_cmds["pos_x"] = fra_cmds["Pos"].apply(extract_x)
    fra_cmds["pos_y"] = fra_cmds["Pos"].apply(extract_y)

    fra_cmds = fra_cmds.dropna(subset=["pos_x", "pos_y"])
    if fra_cmds.empty:
        return result

    # 4) 러시아 방향 두 영역 중 하나라도 포함되는 명령 필터
    x1_min = FRA_ATK_RUS_REGION_1["x_min"]
    x1_max = FRA_ATK_RUS_REGION_1["x_max"]
    y1_min = FRA_ATK_RUS_REGION_1["y_min"]
    y1_max = FRA_ATK_RUS_REGION_1["y_max"]

    x2_min = FRA_ATK_RUS_REGION_2["x_min"]
    x2_max = FRA_ATK_RUS_REGION_2["x_max"]
    y2_min = FRA_ATK_RUS_REGION_2["y_min"]
    y2_max = FRA_ATK_RUS_REGION_2["y_max"]

    in_region1 = (
        (fra_cmds["pos_x"] >= x1_min) & (fra_cmds["pos_x"] <= x1_max) &
        (fra_cmds["pos_y"] >= y1_min) & (fra_cmds["pos_y"] <= y1_max)
    )

    in_region2 = (
        (fra_cmds["pos_x"] >= x2_min) & (fra_cmds["pos_x"] <= x2_max) &
        (fra_cmds["pos_y"] >= y2_min) & (fra_cmds["pos_y"] <= y2_max)
    )

    in_rus = in_region1 | in_region2

    rus_cmds = fra_cmds[in_rus].copy()
    if rus_cmds.empty:
        return result

    cmd_count = int(len(rus_cmds))
    result["fraatkrus_cmd_count"] = cmd_count

    if cmd_count >= 20:
        result["fraatkrus"] = 1

    return result
#프본 침공 여부
def detect_allies_invasion_fra(players_df, cmds_df):
    """
    동맹국(영국, 스페인, 러시아, 신성로마)의 프랑스 본토(파리 인근) 침공 여부를 판정.

    - 영역: ALLIES_ATK_FRA_REGION
        x: 3936 ~ 4640
        y: 3104 ~ 3744

    - 대상 국가:
        gbr, spa, rus, hre (SLOT_TO_NATION 기반으로 PlayerID를 찾음)

    - 판정 기준:
        위 4개국의 명령 중, 해당 영역에 찍힌 명령의 총합이 20회 이상이면
          isparisatked = 1
        그 외에는 0

    - 명령 수는 isparisatked_cmd_count 에 기록.

    반환 dict:
        {
            "isparisatked": 0 또는 1,
            "isparisatked_cmd_count": int (해당 영역 내 명령 총합)
        }
    """
    result = {
        "isparisatked": 0,
        "isparisatked_cmd_count": 0,
    }

    # 기본 체크
    if players_df is None or players_df.empty or cmds_df is None or cmds_df.empty:
        return result

    # --- SLOT_TO_NATION 을 이용해 각 국가 PlayerID 찾기 ---
    slot_to_pid = {}
    for _, row in players_df.iterrows():
        slot = row.get("SlotID")
        pid  = row.get("ID")
        if slot in SLOT_TO_NATION and pid is not None:
            nation = SLOT_TO_NATION[slot]
            slot_to_pid[nation] = pid

    gbr_pid = slot_to_pid.get("gbr")
    spa_pid = slot_to_pid.get("spa")
    rus_pid = slot_to_pid.get("rus")
    hre_pid = slot_to_pid.get("hre")

    allied_pids = [pid for pid in [gbr_pid, spa_pid, rus_pid, hre_pid] if pid is not None]
    if not allied_pids:
        # 4국 모두 없으면 침공 자체가 의미 없음
        return result

    cmds = cmds_df.copy()

    # 좌표 없는 경우 계산 불가
    if "Pos" not in cmds.columns:
        return result

    # --- 동맹국 명령만 추출 ---
    allies_cmds = cmds[cmds["PlayerID"].isin(allied_pids)].copy()
    if allies_cmds.empty:
        return result

    # --- 좌표 추출 ---
    def extract_x(pos):
        if isinstance(pos, dict) and "X" in pos:
            return pos["X"]
        return np.nan

    def extract_y(pos):
        if isinstance(pos, dict) and "Y" in pos:
            return pos["Y"]
        return np.nan

    allies_cmds["pos_x"] = allies_cmds["Pos"].apply(extract_x)
    allies_cmds["pos_y"] = allies_cmds["Pos"].apply(extract_y)
    allies_cmds = allies_cmds.dropna(subset=["pos_x", "pos_y"])
    if allies_cmds.empty:
        return result

    # --- 프랑스 본토(파리 인근) 영역 필터 ---
    x_min = ALLIES_ATK_FRA_REGION["x_min"]
    x_max = ALLIES_ATK_FRA_REGION["x_max"]
    y_min = ALLIES_ATK_FRA_REGION["y_min"]
    y_max = ALLIES_ATK_FRA_REGION["y_max"]

    in_fra_home = (
        (allies_cmds["pos_x"] >= x_min) & (allies_cmds["pos_x"] <= x_max) &
        (allies_cmds["pos_y"] >= y_min) & (allies_cmds["pos_y"] <= y_max)
    )

    fra_home_cmds = allies_cmds[in_fra_home].copy()
    if fra_home_cmds.empty:
        return result

    # --- 명령 수 집계 ---
    cmd_count = int(len(fra_home_cmds))
    result["isparisatked_cmd_count"] = cmd_count

    if cmd_count >= 20:
        result["isparisatked"] = 1

    return result


# ===== 러시아 덴마크/노르웨이 전역 변수 =====

DEN_NOR_REGION1 = {
    "x_min": 4736,
    "x_max": 5376,
    "y_min":  928,
    "y_max": 1984,
}

DEN_NOR_REGION2 = {
    "x_min": 5216,
    "x_max": 5856,
    "y_min":  864,
    "y_max": 1952,
}

DEN_NOR_TIME_WINDOW = (0, 450)  # 0~450초


def infer_rus_den_nor(players_df, cmds_df, denmark_value, fps=FPS):
    """
    덴마크/노르웨이 전역에서 러시아의 공격 여부/미국 방어 여부/노르웨이 점령 여부 추론.

    반환:
      {
        "rusatkden": 0/1,       # 러시아가 해당 지역(두 박스)에 이동/공격 명령을 내렸는지
        "usadefden": 0/1,       # (rusatkden=1일 때) 미국이 해당 지역에 어떤 명령이라도 내렸는지
        "rusconqnor": 0/1,      # 규칙에 따라 러시아가 노르웨이를 점령했는지
        "norwardur": float(초), # 러시아-미국 교전 지속 시간 (명령 공존 구간)
      }

    규칙 요약
    ---------
    - 시간: 0~450초
    - 지역: (4736~5376, 928~1984) ∪ (5216~5856, 864~1952)

    1) rusatkden:
       - 위 시간/지역 안에서 RUS의 이동/공격 명령이 한 번이라도 있으면 1, 아니면 0.
       - rusatkden=0이면 usadefden, rusconqnor, norwardur 모두 0으로 두고 종료.

    2) usadefden:
       - rusatkden=1일 때, 같은 시간/지역 안에서 USA 명령이 한 번이라도 있으면 1, 아니면 0.

    3) norwardur:
       - usadefden=1일 때, 해당 지역에서 RUS와 USA의 명령이 공존한 기간을 “전투”로 보고,
         그 지속 시간(초) = [max(첫 RUS 시간, 첫 USA 시간) ~ min(마지막 RUS 시간, 마지막 USA 시간)]의 길이.
         (겹치는 부분이 없으면 0)

    4) rusconqnor:
       기본값 0에서 아래 조건을 만족하면 1로 설정 (OR 조건), 마지막에 450초 규칙으로 0으로 덮어쓸 수 있음.

       - A) usadefden == 1 이고,
            러시아와 미국 명령이 공존한 마지막 시점 t_last 이후 30초 동안
            해당 지역에 미국/프랑스 명령이 없으면 → rusconqnor = 1

       - B) denmark 이벤트 == 0 이고 usadefden == 0 이면 → rusconqnor = 1

       - C) denmark 이벤트 == 1 이고 프랑스가 해당 지역에 명령을 내린 적이 없으면 → rusconqnor = 1

       - D) (마지막 단계, 우선순위 최상)
            450초에 도달했을 때, 직전 30초(420~450초) 구간 안에
            러시아와 (미국 또는 프랑스)의 명령이 모두 존재하면,
            아직 전투 중이라고 보고 rusconqnor = 0 으로 강제.

    """
    result = {
        "rusatkden": 0,
        "usadefden": 0,
        "rusconqnor": 0,
        "norwardur": 0.0,
    }

    if players_df is None or players_df.empty or cmds_df is None or cmds_df.empty:
        return result

    # --- RUS / USA / FRA PlayerID 찾기 ---
    slot_to_pid = {}
    for _, row in players_df.iterrows():
        slot = row.get("SlotID")
        pid  = row.get("ID")
        if slot in SLOT_TO_NATION and pid is not None:
            slot_to_pid[SLOT_TO_NATION[slot]] = pid

    rus_pid = slot_to_pid.get("rus")
    usa_pid = slot_to_pid.get("usa")
    fra_pid = slot_to_pid.get("fra")

    if rus_pid is None:
        # 러시아가 없으면 이 전역은 의미 없음
        return result

    cmds = cmds_df.copy()

    # --- 명령 이름 통합 ---
    if "Order" in cmds.columns:
        cmds["order_name"] = cmds["Order"].apply(_extract_name_from_field)
    else:
        cmds["order_name"] = None

    if "Type" in cmds.columns:
        cmds["type_name"] = cmds["Type"].apply(_extract_name_from_field)
    else:
        cmds["type_name"] = None

    cmds["cmd_name"]   = cmds["order_name"].fillna(cmds["type_name"]).astype(str)
    cmds["cmd_name_l"] = cmds["cmd_name"].str.lower()

    # --- 좌표 추출 ---
    def extract_x(pos):
        if isinstance(pos, dict) and "X" in pos:
            return pos["X"]
        return np.nan

    def extract_y(pos):
        if isinstance(pos, dict) and "Y" in pos:
            return pos["Y"]
        return np.nan

    cmds["pos_x"] = cmds["Pos"].apply(extract_x)
    cmds["pos_y"] = cmds["Pos"].apply(extract_y)

    # --- 지역/시간 마스크 ---
    def in_region(df, region):
        return (
            (df["pos_x"] >= region["x_min"]) & (df["pos_x"] <= region["x_max"]) &
            (df["pos_y"] >= region["y_min"]) & (df["pos_y"] <= region["y_max"])
        )

    in_reg1 = in_region(cmds, DEN_NOR_REGION1)
    in_reg2 = in_region(cmds, DEN_NOR_REGION2)
    in_den_nor_region = in_reg1 | in_reg2

    start_sec, end_sec = DEN_NOR_TIME_WINDOW
    start_frame = int(start_sec * fps)
    end_frame   = int(end_sec   * fps)

    mask_time = (cmds["Frame"] >= start_frame) & (cmds["Frame"] <= end_frame)

    # 이동/공격 명령 필터
    move_names = {"right click", "move", "patrol"}
    is_attack = cmds["cmd_name_l"].str.contains("attack")   # attack / attackmove / attack1/2 포함
    is_move   = cmds["cmd_name_l"].isin(move_names)
    is_act    = is_attack | is_move

    mask_region = in_den_nor_region & mask_time & is_act
    if not mask_region.any():
        return result  # 해당 전역에 명령이 아예 없음

    region_cmds = cmds[mask_region].copy()

    # --- 러시아 명령 여부: rusatkden ---
    rus_cmds = region_cmds[region_cmds["PlayerID"] == rus_pid].copy()
    if rus_cmds.shape[0] < 4:
        # 러시아 명령이 4개 미만이면 공격으로 보지 않음
        return result  # rusatkden=0, usadefden/rusconqnor/norwardur도 0 유지

    result["rusatkden"] = 1

    # --- 미국/프랑스 명령 ---
    if usa_pid is not None:
        usa_cmds = region_cmds[region_cmds["PlayerID"] == usa_pid].copy()
    else:
        usa_cmds = region_cmds.iloc[0:0].copy()

    if fra_pid is not None:
        fra_cmds = region_cmds[region_cmds["PlayerID"] == fra_pid].copy()
    else:
        fra_cmds = region_cmds.iloc[0:0].copy()

    if not usa_cmds.empty:
        result["usadefden"] = 1
    else:
        result["usadefden"] = 0

    fra_in_region = not fra_cmds.empty

    # --- norwardur (러시아-미국 공존 시간) 계산 ---
    last_coexist_time = None
    if result["usadefden"] == 1:
        rus_times = np.sort(rus_cmds["Frame"].values.astype(float) / fps)
        usa_times = np.sort(usa_cmds["Frame"].values.astype(float) / fps)
        if rus_times.size > 0 and usa_times.size > 0:
            start_overlap = max(rus_times[0], usa_times[0])
            end_overlap   = min(rus_times[-1], usa_times[-1])
            if end_overlap > start_overlap:
                result["norwardur"] = float(end_overlap - start_overlap)
                last_coexist_time = float(end_overlap)

    # --- rusconqnor 기본값 ---
    rusconqnor = 0

    # B) denmark==0 이고 usadefden==0 이면 러시아 점령
    if (denmark_value == 0) and (result["usadefden"] == 0):
        rusconqnor = 1

    # C) denmark==1 이고 프랑스가 해당 지역에 명령을 내린 적이 없으면 러시아 점령
    if (denmark_value == 1) and (not fra_in_region):
        rusconqnor = 1
    
    if (
      result.get("rusatkden", 0) == 1 and
      result["usadefden"] == 0 and
      denmark_value == 1 and
      result["norwardur"] == 0
      ):
      rusconqnor = 1
    if (
       result.get("rusatkden", 0) == 1 and
       result["usadefden"] == 0 and
       denmark_value == 2 and
       result["norwardur"] == 0
       ):
       rusconqnor = 1

    # A) usadefden==1 이고, 마지막 공존 시점 이후 30초 동안 미국/프랑스 명령이 없으면 러시아 점령
    defender_pids = [pid for pid in (usa_pid, fra_pid) if pid is not None]
    if (result["usadefden"] == 1) and (last_coexist_time is not None) and defender_pids:
        win_window_start = int(last_coexist_time * fps)
        win_window_end   = int(min(last_coexist_time + 30.0, end_sec) * fps)

        mask_def_after = (
            (cmds["Frame"] > win_window_start) &
            (cmds["Frame"] <= win_window_end) &
            in_den_nor_region &
            (cmds["PlayerID"].isin(defender_pids))
        )
        if not mask_def_after.any():
            rusconqnor = 1

    # D) 450초에 도달했을 때, 러시아와 미국/프랑스가 여전히 해당 지역에서 전투 중이면 패배로 덮어쓰기
    if defender_pids:
        T = end_sec  # 450
        T_frame = int(T * fps)
        recent_start = int(max(start_sec, T - 30.0) * fps)  # 직전 30초

        mask_rus_recent = (
            (cmds["PlayerID"] == rus_pid) &
            in_den_nor_region &
            (cmds["Frame"] >= recent_start) &
            (cmds["Frame"] <= T_frame)
        )
        mask_def_recent = (
            cmds["PlayerID"].isin(defender_pids) &
            in_den_nor_region &
            (cmds["Frame"] >= recent_start) &
            (cmds["Frame"] <= T_frame)
        )

        if mask_rus_recent.any() and mask_def_recent.any():
            rusconqnor = 0

    result["rusconqnor"] = int(rusconqnor)
    return result

def turatkind(players_df, cmds_df):
    """
    오스만이 인도 지역(7936~8192, 4928~5248)에 0~600초 사이에
    공격/이동했는지 판정.

    - turatkindia = 1 if
        1) 해당 지역에 moveunload 계열 커맨드를 한 번이라도 내렸거나
        2) 해당 지역에 이동/이동공격(Right Click, Move, AttackMove, Patrol, Attack1, Attack2)
           계열 커맨드를 3회 이상 내렸을 때
      그 외에는 0

    추가:
      - turlastseize : 0~600초 구간에서 오스만이 해당 지역에 마지막으로 명령을
        내린 시각(초, Frame/FPS). 명령이 없으면 NaN.

    디버깅용:
      - tur_cmd_cnt_india          : 해당 지역에 내려진 오스만의 모든 명령 수
      - tur_moveunload_cnt_india   : 지역 내 unload 계열 명령 수
      - tur_moveatk_cnt_india      : 지역 내 이동/이동공격 계열 명령 수
    """
    result = {
        "turatkindia": 0,
        "tur_cmd_cnt_india": 0,            # ✅ 총 커맨드 수
        "tur_moveunload_cnt_india": 0,
        "tur_moveatk_cnt_india": 0,
        "turlastseize": np.nan,
    }

    if cmds_df is None or cmds_df.empty or players_df is None or players_df.empty:
        return result

    # 1) 오스만 PlayerID 찾기 (슬롯 7 = tur 전제)
    try:
        tur_pid = players_df.loc[players_df["SlotID"] == 7, "ID"].iloc[0]
    except IndexError:
        return result

    cmds = cmds_df.copy()

    # Frame이 없으면 시간 필터를 못하니 바로 반환
    if "Frame" not in cmds.columns:
        return result

    # 2) 시간창(0 ~ 600초) → 프레임으로 변환 후 필터
    start_sec, end_sec = TUR_ATK_INDIA_TIME_WINDOW
    start_frame = int(start_sec * FPS)
    end_frame   = int(end_sec * FPS)

    time_mask = (cmds["Frame"] >= start_frame) & (cmds["Frame"] <= end_frame)
    cmds = cmds[time_mask].copy()
    if cmds.empty:
        return result

    # 3) Order.Name / Type.Name 통합해서 cmd_name 만들기 (소문자)
    if "Order" in cmds.columns:
        cmds["order_name"] = cmds["Order"].apply(_extract_name_from_field)
    else:
        cmds["order_name"] = None

    if "Type" in cmds.columns:
        cmds["type_name"] = cmds["Type"].apply(_extract_name_from_field)
    else:
        cmds["type_name"] = None

    cmds["cmd_name"] = cmds["order_name"].fillna(cmds["type_name"]).astype(str).str.lower()

    # 4) 오스만 + 좌표 있는 명령만 사용
    tur_cmds = cmds[cmds["PlayerID"] == tur_pid].copy()
    if tur_cmds.empty:
        return result

    def extract_x(pos):
        if isinstance(pos, dict) and "X" in pos:
            return pos["X"]
        return np.nan

    def extract_y(pos):
        if isinstance(pos, dict) and "Y" in pos:
            return pos["Y"]
        return np.nan

    if "Pos" in tur_cmds.columns:
        tur_cmds["pos_x"] = tur_cmds["Pos"].apply(extract_x)
        tur_cmds["pos_y"] = tur_cmds["Pos"].apply(extract_y)
    else:
        tur_cmds["pos_x"] = np.nan
        tur_cmds["pos_y"] = np.nan

    tur_cmds = tur_cmds.dropna(subset=["pos_x", "pos_y"])
    if tur_cmds.empty:
        return result

    # 5) 인도 지역 박스 안 명령만 필터
    x_min = TUR_ATK_INDIA_REGION["x_min"]
    x_max = TUR_ATK_INDIA_REGION["x_max"]
    y_min = TUR_ATK_INDIA_REGION["y_min"]
    y_max = TUR_ATK_INDIA_REGION["y_max"]

    in_region = (
        (tur_cmds["pos_x"] >= x_min) & (tur_cmds["pos_x"] <= x_max) &
        (tur_cmds["pos_y"] >= y_min) & (tur_cmds["pos_y"] <= y_max)
    )

    region_cmds = tur_cmds[in_region].copy()
    if region_cmds.empty:
        # 0~600초 사이에 인도 지역 명령이 전혀 없으면 그대로 0/NaN
        return result

    # ✅ 인도 지역에서의 오스만 전체 명령 수
    result["tur_cmd_cnt_india"] = int(region_cmds.shape[0])

    # 6) moveunload 계열: 이름에 "unload"가 들어간 명령
    mask_moveunload = region_cmds["cmd_name"].str.contains("unload")
    moveunload_count = int(mask_moveunload.sum())
    result["tur_moveunload_cnt_india"] = moveunload_count

    # 7) 이동/이동공격 계열: Right Click, Move, AttackMove, Patrol, Attack1, Attack2
    moveatk_names = {
        "right click",
        "move",
        "attackmove",
        "patrol",
        "attack1",
        "attack2",
    }
    mask_moveatk = region_cmds["cmd_name"].isin(moveatk_names)
    moveatk_count = int(mask_moveatk.sum())
    result["tur_moveatk_cnt_india"] = moveatk_count

    # 8) 최종 판정 + turlastseize 계산
    if (moveunload_count > 0) or (moveatk_count >= 3):
        # 조건을 만족하는 경우에만 1로 세팅 + 마지막 시각 기록
        result["turatkindia"] = 1

        # 마지막 명령 시각 (Frame → 초)
        last_frame = int(region_cmds["Frame"].max())
        result["turlastseize"] = last_frame / FPS
    else:
        # 조건을 만족하지 못하면 turatkindia=0, turlastseize는 NaN 유지
        result["turatkindia"] = 0
        result["turlastseize"] = np.nan

    return result


# === 미국 캐나다 공격 판정용 영역/시간 설정 ===

USA_ATK_CAN_REGION = {
    "x_min": 0,
    "x_max": 2336,
    "y_min": 2560,
    "y_max": 3296,
}

# 0초 ~ 900초 동안만 캐나다 공격 시도 체크
USA_ATK_CAN_TIME_WINDOW = (0, 900)  # [sec]

#미국 캐나다 침공
def usaatkcan(players_df, cmds_df):
    """
    미국이 캐나다 지역(USA_ATK_CAN_REGION)에 공격 명령을 내렸는지 확인하는 함수.

    - usaatkcan      : 캐나다 지역에 한 번이라도 공격 계열 명령이 있으면 1, 아니면 0
    - usaatkcan_count: 해당 지역에 내려진 공격 계열 명령의 개수

    공격 계열 명령은 Order/Type 이름에 'attack' 이 포함된 커맨드로 정의.
    시간 범위는 USA_ATK_CAN_TIME_WINDOW (기본 0~600초).
    """
    result = {
        "usaatkcan": np.nan,
        "usaatkcan_count": np.nan,
    }

    if cmds_df is None or cmds_df.empty or players_df is None or players_df.empty:
        # 데이터 자체가 없으면 0으로 처리해도 되고, 지금처럼 NaN 유지해도 됨
        result["usaatkcan"] = 0
        result["usaatkcan_count"] = 0
        return result

    # 1) 미국 PlayerID 찾기 (슬롯 3 = usa 전제)
    try:
        usa_pid = players_df.loc[players_df["SlotID"] == 3, "ID"].iloc[0]
    except IndexError:
        result["usaatkcan"] = 0
        result["usaatkcan_count"] = 0
        return result

    # 2) 시간창 → 프레임
    start_sec, end_sec = USA_ATK_CAN_TIME_WINDOW
    start_frame = int(start_sec * FPS)
    end_frame   = int(end_sec * FPS)

    cmds = cmds_df.copy()

    # 3) Order.Name / Type.Name 통합해서 cmd_name 만들기
    if "Order" in cmds.columns:
        cmds["order_name"] = cmds["Order"].apply(_extract_name_from_field)
    else:
        cmds["order_name"] = None

    if "Type" in cmds.columns:
        cmds["type_name"] = cmds["Type"].apply(_extract_name_from_field)
    else:
        cmds["type_name"] = None

    # Order 우선, 없으면 Type 사용
    cmds["cmd_name"] = cmds["order_name"].fillna(cmds["type_name"]).astype(str).str.lower()

    # 4) 미국 + 시간창 + 'attack'이 들어가는 명령만 필터
    mask_player = cmds["PlayerID"] == usa_pid
    mask_time   = (cmds["Frame"] >= start_frame) & (cmds["Frame"] <= end_frame)
    mask_attack = cmds["cmd_name"].str.contains("attack")

    candidate = cmds[mask_player & mask_time & mask_attack].copy()
    if candidate.empty:
        result["usaatkcan"] = 0
        result["usaatkcan_count"] = 0
        return result

    # 5) 좌표 추출
    def extract_x(pos):
        if isinstance(pos, dict) and "X" in pos:
            return pos["X"]
        return np.nan

    def extract_y(pos):
        if isinstance(pos, dict) and "Y" in pos:
            return pos["Y"]
        return np.nan

    candidate["pos_x"] = candidate["Pos"].apply(extract_x)
    candidate["pos_y"] = candidate["Pos"].apply(extract_y)
    candidate = candidate.dropna(subset=["pos_x", "pos_y"])
    if candidate.empty:
        result["usaatkcan"] = 0
        result["usaatkcan_count"] = 0
        return result

    # 6) 캐나다 박스 안 공격 명령 수 세기
    cx_min = USA_ATK_CAN_REGION["x_min"]
    cx_max = USA_ATK_CAN_REGION["x_max"]
    cy_min = USA_ATK_CAN_REGION["y_min"]
    cy_max = USA_ATK_CAN_REGION["y_max"]

    in_can = (
        (candidate["pos_x"] >= cx_min) & (candidate["pos_x"] <= cx_max) &
        (candidate["pos_y"] >= cy_min) & (candidate["pos_y"] <= cy_max)
    )

    can_count = int(in_can.sum())
    result["usaatkcan_count"] = can_count
    result["usaatkcan"] = 1 if can_count > 0 else 0

    return result

#러시아 중동공격
def detect_rus_invasion_tur(players_df, cmds_df):
    """
    러시아의 아나톨리아/캅카스&페르시아 침공 여부를 판정.

    - 좌표:
        # 아나톨리아 : x 6208~7072, y 3808~4480
        # 캅카스&페르시아 : x 7008~8192, y 3648~4992

    - rusfastinv (0~300초):
        0~300초 사이에 위 두 지역 중 어느 한 곳이라도
        러시아의 공격/이동 계열 명령이 합계 2회 이상이면 1, 아니면 0

    - rusnorinv (301초 이후):
        301초 이후(301초 * FPS 프레임 이상) 위 두 지역에서
        러시아의 공격/이동 계열 명령이 합계 2회 이상이면 1, 아니면 0

    - 둘 다 기본값은 0. 러시아 명령이 아예 없으면 둘 다 0.
    """

    result = {
        "rusfastinv": 0,
        "rusnorinv": 0,
    }

    # 기본 체크
    if cmds_df is None or cmds_df.empty or players_df is None or players_df.empty:
        return result

    # 1) 러시아 PlayerID 찾기 (SLOT_TO_NATION에서 rus는 SlotID=1로 가정)
    try:
        rus_pid = players_df.loc[players_df["SlotID"] == 1, "ID"].iloc[0]
    except IndexError:
        return result

    cmds = cmds_df.copy()

    # 2) Order.Name / Type.Name → cmd_name 통합
    if "Order" in cmds.columns:
        cmds["order_name"] = cmds["Order"].apply(_extract_name_from_field)
    else:
        cmds["order_name"] = None

    if "Type" in cmds.columns:
        cmds["type_name"] = cmds["Type"].apply(_extract_name_from_field)
    else:
        cmds["type_name"] = None

    cmds["cmd_name"] = cmds["order_name"].fillna(cmds["type_name"]).astype(str).str.lower()

    # 3) 러시아 + 공격/이동 계열 명령만 필터
    mask_player = cmds["PlayerID"] == rus_pid

    # 기존 MOVE_CMD_NAMES는 대문자/혼합이라, 여기서 소문자로 맞춤
    move_names_lower = {name.lower() for name in MOVE_CMD_NAMES}

    def is_move_or_attack(name: str) -> bool:
        if not isinstance(name, str):
            return False
        if "attack" in name:  # attack, attackmove, attack1, attack2 전부 포함
            return True
        return name in move_names_lower  # right click, move, patrol 등

    mask_action = cmds["cmd_name"].apply(is_move_or_attack)

    cmds = cmds[mask_player & mask_action].copy()
    if cmds.empty:
        return result

    # 4) 좌표 추출
    def extract_x(pos):
        if isinstance(pos, dict) and "X" in pos:
            return pos["X"]
        return np.nan

    def extract_y(pos):
        if isinstance(pos, dict) and "Y" in pos:
            return pos["Y"]
        return np.nan

    cmds["pos_x"] = cmds["Pos"].apply(extract_x)
    cmds["pos_y"] = cmds["Pos"].apply(extract_y)
    cmds = cmds.dropna(subset=["pos_x", "pos_y"]).copy()
    if cmds.empty:
        return result

    # 5) 아나톨리아 / 캅카스&페르시아 지역 정의
    anatolia = {
        "x_min": 6208,
        "x_max": 7072,
        "y_min": 3808,
        "y_max": 4480,
    }
    cauc_persia = {
        "x_min": 7008,
        "x_max": 8192,
        "y_min": 3648,
        "y_max": 4992,
    }

    def in_region(df, reg):
        return (
            (df["pos_x"] >= reg["x_min"]) & (df["pos_x"] <= reg["x_max"]) &
            (df["pos_y"] >= reg["y_min"]) & (df["pos_y"] <= reg["y_max"])
        )

    mask_region = in_region(cmds, anatolia) | in_region(cmds, cauc_persia)
    cmds = cmds[mask_region].copy()
    if cmds.empty:
        # 두 지역 안에 들어온 명령이 없으면 침공 없는 것으로 간주
        return result

    # 6) 시간 구간 나누기: 0~300초, 301초~
    fast_end_frame = int(300 * FPS)
    late_start_frame = int(301 * FPS)

    fast_mask = (cmds["Frame"] <= fast_end_frame)
    late_mask = (cmds["Frame"] >= late_start_frame)

    fast_count = int(fast_mask.sum())
    late_count = int(late_mask.sum())

    if fast_count >= 2:
        result["rusfastinv"] = 1

    if late_count >= 2:
        result["rusnorinv"] = 1

    return result
#영국이탈 상륙
def detect_gbr_invasion_italy(players_df, cmds_df):
    """
    영국의 이탈리아 상륙 여부 판정.

    - 좌표(이탈리아 전역):
        x: 4512 ~ 5184
        y: 3456 ~ 4416

    - 조건:
        영국(PlayerID = SlotID 0)이 위 지역에
        'unload' 계열(moveunload 포함) 명령을 2번 이상 내렸으면 gbrinvita = 1
        그렇지 않으면 0

    반환:
        {
            "gbrinvita": 0 또는 1,
            "gbr_moveunload_cnt_italy": 이탈리아 지역 내 unload 계열 명령 수
        }
    """

    result = {
        "gbrinvita": 0,
        "gbr_moveunload_cnt_italy": 0,   # ✅ 언로드 커맨드 개수
    }

    if cmds_df is None or cmds_df.empty or players_df is None or players_df.empty:
        return result

    # 1) 영국 PlayerID 찾기 (슬롯 0 = gbr 전제)
    try:
        gbr_pid = players_df.loc[players_df["SlotID"] == 0, "ID"].iloc[0]
    except IndexError:
        return result

    cmds = cmds_df.copy()

    # 2) Order.Name / Type.Name 통합해서 cmd_name 만들기
    if "Order" in cmds.columns:
        cmds["order_name"] = cmds["Order"].apply(_extract_name_from_field)
    else:
        cmds["order_name"] = None

    if "Type" in cmds.columns:
        cmds["type_name"] = cmds["Type"].apply(_extract_name_from_field)
    else:
        cmds["type_name"] = None

    # Order 우선, 없으면 Type 사용
    cmds["cmd_name"] = cmds["order_name"].fillna(cmds["type_name"]).astype(str).str.lower()

    # 3) 영국 + 'unload' 계열 명령만 필터 (moveunload 포함)
    mask_player = cmds["PlayerID"] == gbr_pid
    mask_unload = cmds["cmd_name"].str.contains("unload")  # moveunload, unload 모두 포함

    candidate = cmds[mask_player & mask_unload].copy()
    if candidate.empty:
        return result

    # 4) 좌표 추출
    def extract_x(pos):
        if isinstance(pos, dict) and "X" in pos:
            return pos["X"]
        return np.nan

    def extract_y(pos):
        if isinstance(pos, dict) and "Y" in pos:
            return pos["Y"]
        return np.nan

    candidate["pos_x"] = candidate["Pos"].apply(extract_x)
    candidate["pos_y"] = candidate["Pos"].apply(extract_y)
    candidate = candidate.dropna(subset=["pos_x", "pos_y"]).copy()
    if candidate.empty:
        return result

    # 5) 이탈리아 영역 정의
    ITA_REGION = {
        "x_min": 4512,
        "x_max": 5184,
        "y_min": 3456,
        "y_max": 4416,
    }

    in_ita = (
        (candidate["pos_x"] >= ITA_REGION["x_min"]) &
        (candidate["pos_x"] <= ITA_REGION["x_max"]) &
        (candidate["pos_y"] >= ITA_REGION["y_min"]) &
        (candidate["pos_y"] <= ITA_REGION["y_max"])
    )

    ita_unload_count = int(in_ita.sum())

    # ✅ 언로드 커맨드 수 기록
    result["gbr_moveunload_cnt_italy"] = ita_unload_count

    # 상륙 여부 플래그
    if ita_unload_count >= 2:
        result["gbrinvita"] = 1

    return result
#오스만이집트
def turatkegypt(players_df, cmds_df):
    """
    오스만의 이집트 공격 여부 및 공격 명령 수를 계산.

    - 이집트 좌표:
        x: 5540 ~ 6240
        y: 4704 ~ 5344

    - 시간: 0 ~ 600초 (TUR_ATK_EGYPT_TIME_WINDOW 사용)

    - 공격/공격이동 계열:
        cmd_name 에 "attack" 이 들어가는 명령들
        (Attack, AttackMove, Attack1, Attack2 등)

    반환 dict:
        {
            "turatkegypt": 0 또는 1,
            "turatkegypt_cmd_count": 지역 내 공격/공격이동 명령 수 (int)
        }
    """
    result = {
        "turatkegypt": 0,
        "turatkegypt_cmd_count": 0,
    }

    if cmds_df is None or cmds_df.empty or players_df is None or players_df.empty:
        return result

    # 1) 오스만 PlayerID 찾기 (슬롯 7 = tur 전제)
    try:
        tur_pid = players_df.loc[players_df["SlotID"] == 7, "ID"].iloc[0]
    except IndexError:
        return result

    cmds = cmds_df.copy()

    # Frame 없으면 시간 필터 불가
    if "Frame" not in cmds.columns:
        return result

    # 2) 시간창 필터링 (0~600초)
    start_sec, end_sec = TUR_ATK_EGYPT_TIME_WINDOW
    start_frame = int(start_sec * FPS)
    end_frame   = int(end_sec * FPS)

    time_mask = (cmds["Frame"] >= start_frame) & (cmds["Frame"] <= end_frame)
    cmds = cmds[time_mask].copy()
    if cmds.empty:
        return result

    # 3) Order.Name / Type.Name 통합 → cmd_name
    if "Order" in cmds.columns:
        cmds["order_name"] = cmds["Order"].apply(_extract_name_from_field)
    else:
        cmds["order_name"] = None

    if "Type" in cmds.columns:
        cmds["type_name"] = cmds["Type"].apply(_extract_name_from_field)
    else:
        cmds["type_name"] = None

    cmds["cmd_name"] = cmds["order_name"].fillna(cmds["type_name"]).astype(str).str.lower()

    # 4) 오스만 명령만
    tur_cmds = cmds[cmds["PlayerID"] == tur_pid].copy()
    if tur_cmds.empty:
        return result

    # 5) 좌표 추출
    def extract_x(pos):
        if isinstance(pos, dict) and "X" in pos:
            return pos["X"]
        return np.nan

    def extract_y(pos):
        if isinstance(pos, dict) and "Y" in pos:
            return pos["Y"]
        return np.nan

    if "Pos" in tur_cmds.columns:
        tur_cmds["pos_x"] = tur_cmds["Pos"].apply(extract_x)
        tur_cmds["pos_y"] = tur_cmds["Pos"].apply(extract_y)
    else:
        tur_cmds["pos_x"] = np.nan
        tur_cmds["pos_y"] = np.nan

    tur_cmds = tur_cmds.dropna(subset=["pos_x", "pos_y"])
    if tur_cmds.empty:
        return result

    # 6) 이집트 영역 안 명령만 필터
    x_min = TUR_ATK_EGYPT_REGION["x_min"]
    x_max = TUR_ATK_EGYPT_REGION["x_max"]
    y_min = TUR_ATK_EGYPT_REGION["y_min"]
    y_max = TUR_ATK_EGYPT_REGION["y_max"]

    in_egypt = (
        (tur_cmds["pos_x"] >= x_min) & (tur_cmds["pos_x"] <= x_max) &
        (tur_cmds["pos_y"] >= y_min) & (tur_cmds["pos_y"] <= y_max)
    )

    egypt_cmds = tur_cmds[in_egypt].copy()
    if egypt_cmds.empty:
        return result

    # 7) 공격/공격이동 계열만 선택: 이름에 'attack' 이 들어가는 명령
    mask_attack = egypt_cmds["cmd_name"].str.contains("attack")
    atk_cmds = egypt_cmds[mask_attack].copy()

    atk_count = int(mask_attack.sum())
    result["turatkegypt_cmd_count"] = atk_count

    if atk_count > 0:
        result["turatkegypt"] = 1

    return result
#프랑스 남미
def fraatksam(players_df, cmds_df):
    """
    프랑스의 남미(사우스 아메리카) 공격 여부 및 공격 명령 수를 계산.

    - 남미 좌표:
        x: 1344 ~ 2976
        y: 5440 ~ 7776

    - 시간: 0 ~ 600초 (FRA_ATK_SAM_TIME_WINDOW 사용)

    - 공격/공격이동 계열:
        cmd_name 에 "attack" 이 들어가는 명령들
        (Attack, AttackMove, Attack1, Attack2 등)

    반환 dict:
        {
            "fraatksam": 0 또는 1,
            "fraatksam_cmd_count": 남미 지역 내 공격/공격이동 명령 수 (int)
        }
    """
    result = {
        "fraatksam": 0,
        "fraatksam_cmd_count": 0,
    }

    if cmds_df is None or cmds_df.empty or players_df is None or players_df.empty:
        return result

    # 1) 프랑스 PlayerID 찾기 (슬롯 2 = fra 전제)
    try:
        fra_pid = players_df.loc[players_df["SlotID"] == 2, "ID"].iloc[0]
    except IndexError:
        return result

    cmds = cmds_df.copy()

    # Frame 없으면 시간 필터 불가
    if "Frame" not in cmds.columns:
        return result

    # 2) 시간창 필터링 (0~600초)
    start_sec, end_sec = FRA_ATK_SAM_TIME_WINDOW
    start_frame = int(start_sec * FPS)
    end_frame   = int(end_sec * FPS)

    time_mask = (cmds["Frame"] >= start_frame) & (cmds["Frame"] <= end_frame)
    cmds = cmds[time_mask].copy()
    if cmds.empty:
        return result

    # 3) Order.Name / Type.Name 통합 → cmd_name
    if "Order" in cmds.columns:
        cmds["order_name"] = cmds["Order"].apply(_extract_name_from_field)
    else:
        cmds["order_name"] = None

    if "Type" in cmds.columns:
        cmds["type_name"] = cmds["Type"].apply(_extract_name_from_field)
    else:
        cmds["type_name"] = None

    cmds["cmd_name"] = cmds["order_name"].fillna(cmds["type_name"]).astype(str).str.lower()

    # 4) 프랑스 명령만
    fra_cmds = cmds[cmds["PlayerID"] == fra_pid].copy()
    if fra_cmds.empty:
        return result

    # 5) 좌표 추출
    def extract_x(pos):
        if isinstance(pos, dict) and "X" in pos:
            return pos["X"]
        return np.nan

    def extract_y(pos):
        if isinstance(pos, dict) and "Y" in pos:
            return pos["Y"]
        return np.nan

    if "Pos" in fra_cmds.columns:
        fra_cmds["pos_x"] = fra_cmds["Pos"].apply(extract_x)
        fra_cmds["pos_y"] = fra_cmds["Pos"].apply(extract_y)
    else:
        fra_cmds["pos_x"] = np.nan
        fra_cmds["pos_y"] = np.nan

    fra_cmds = fra_cmds.dropna(subset=["pos_x", "pos_y"])
    if fra_cmds.empty:
        return result

    # 6) 남미 영역 안 명령만 필터
    x_min = FRA_ATK_SAM_REGION["x_min"]
    x_max = FRA_ATK_SAM_REGION["x_max"]
    y_min = FRA_ATK_SAM_REGION["y_min"]
    y_max = FRA_ATK_SAM_REGION["y_max"]

    in_sam = (
        (fra_cmds["pos_x"] >= x_min) & (fra_cmds["pos_x"] <= x_max) &
        (fra_cmds["pos_y"] >= y_min) & (fra_cmds["pos_y"] <= y_max)
    )

    sam_cmds = fra_cmds[in_sam].copy()
    if sam_cmds.empty:
        return result

    # 7) 공격/공격이동 계열만 선택: 이름에 'attack' 이 들어가는 명령
    mask_attack = sam_cmds["cmd_name"].str.contains("attack")
    atk_cmds = sam_cmds[mask_attack].copy()

    atk_count = int(mask_attack.sum())
    result["fraatksam_cmd_count"] = atk_count

    if atk_count > 0:
        result["fraatksam"] = 1

    return result
#미국 멕시코
def usaatkmex(players_df, cmds_df):
    """
    미국의 멕시코(북멕시코 전역) 공격 여부 및 공격 명령 수를 계산.

    - 멕시코 좌표:
        x: 0 ~ 768
        y: 4480 ~ 5248

    - 시간: 0 ~ 600초 (USA_ATK_MEX_TIME_WINDOW 사용)

    - 공격/공격이동 계열:
        cmd_name 에 "attack" 이 들어가는 명령들
        (Attack, AttackMove, Attack1, Attack2 등)

    반환 dict:
        {
            "usaatkmex": 0 또는 1,
            "usaatkmex_cmd_count": 멕시코 지역 내 공격/공격이동 명령 수 (int)
        }
    """
    result = {
        "usaatkmex": 0,
        "usaatkmex_cmd_count": 0,
    }

    if cmds_df is None or cmds_df.empty or players_df is None or players_df.empty:
        return result

    # 1) 미국 PlayerID 찾기 (슬롯 3 = usa 전제)
    try:
        usa_pid = players_df.loc[players_df["SlotID"] == 3, "ID"].iloc[0]
    except IndexError:
        return result

    cmds = cmds_df.copy()

    # Frame 없으면 시간 필터 불가
    if "Frame" not in cmds.columns:
        return result

    # 2) 시간창 필터링 (0~600초)
    start_sec, end_sec = USA_ATK_MEX_TIME_WINDOW
    start_frame = int(start_sec * FPS)
    end_frame   = int(end_sec * FPS)

    time_mask = (cmds["Frame"] >= start_frame) & (cmds["Frame"] <= end_frame)
    cmds = cmds[time_mask].copy()
    if cmds.empty:
        return result

    # 3) Order.Name / Type.Name 통합 → cmd_name
    if "Order" in cmds.columns:
        cmds["order_name"] = cmds["Order"].apply(_extract_name_from_field)
    else:
        cmds["order_name"] = None

    if "Type" in cmds.columns:
        cmds["type_name"] = cmds["Type"].apply(_extract_name_from_field)
    else:
        cmds["type_name"] = None

    cmds["cmd_name"] = cmds["order_name"].fillna(cmds["type_name"]).astype(str).str.lower()

    # 4) 미국 명령만
    usa_cmds = cmds[cmds["PlayerID"] == usa_pid].copy()
    if usa_cmds.empty:
        return result

    # 5) 좌표 추출
    def extract_x(pos):
        if isinstance(pos, dict) and "X" in pos:
            return pos["X"]
        return np.nan

    def extract_y(pos):
        if isinstance(pos, dict) and "Y" in pos:
            return pos["Y"]
        return np.nan

    if "Pos" in usa_cmds.columns:
        usa_cmds["pos_x"] = usa_cmds["Pos"].apply(extract_x)
        usa_cmds["pos_y"] = usa_cmds["Pos"].apply(extract_y)
    else:
        usa_cmds["pos_x"] = np.nan
        usa_cmds["pos_y"] = np.nan

    usa_cmds = usa_cmds.dropna(subset=["pos_x", "pos_y"])
    if usa_cmds.empty:
        return result

    # 6) 멕시코 영역 안 명령만 필터
    x_min = USA_ATK_MEX_REGION["x_min"]
    x_max = USA_ATK_MEX_REGION["x_max"]
    y_min = USA_ATK_MEX_REGION["y_min"]
    y_max = USA_ATK_MEX_REGION["y_max"]

    in_mex = (
        (usa_cmds["pos_x"] >= x_min) & (usa_cmds["pos_x"] <= x_max) &
        (usa_cmds["pos_y"] >= y_min) & (usa_cmds["pos_y"] <= y_max)
    )

    mex_cmds = usa_cmds[in_mex].copy()
    if mex_cmds.empty:
        return result

    # 7) 공격/공격이동 계열만 선택: 이름에 'attack' 이 들어가는 명령
    mask_attack = mex_cmds["cmd_name"].str.contains("attack")
    atk_cmds = mex_cmds[mask_attack].copy()

    atk_count = int(mask_attack.sum())
    result["usaatkmex_cmd_count"] = atk_count

    if atk_count > 0:
        result["usaatkmex"] = 1

    return result
#스페인 미국
def spaatkusa(players_df, cmds_df):
    """
    스페인이 미국 본토 지역(SPA_ATK_USA_REGION)에
    공격 계열 명령을 내렸는지 확인하는 함수.

    - spaatkusa : 미국 본토 박스 안에서 한 번이라도 공격 계열 명령이 있으면 1, 아니면 0

    공격 계열 명령은 Order/Type 이름에 'attack' 이 포함된 커맨드로 정의.
    시간 범위는 SPA_ATK_USA_TIME_WINDOW (기본 0~900초).
    """
    result = {
        "spaatkusa": np.nan,
    }

    if cmds_df is None or cmds_df.empty or players_df is None or players_df.empty:
        return result

    # 1) 스페인 PlayerID 찾기 (SlotID = 4)
    try:
        spa_pid = players_df.loc[players_df["SlotID"] == 4, "ID"].iloc[0]
    except IndexError:
        # 스페인이 없는 경기
        result["spaatkusa"] = 0
        return result

    cmds = cmds_df.copy()

    # 2) Order.Name / Type.Name 통합해서 cmd_name 만들기
    if "Order" in cmds.columns:
        cmds["order_name"] = cmds["Order"].apply(_extract_name_from_field)
    else:
        cmds["order_name"] = None

    if "Type" in cmds.columns:
        cmds["type_name"] = cmds["Type"].apply(_extract_name_from_field)
    else:
        cmds["type_name"] = None

    # Order 우선, 없으면 Type 사용
    cmds["cmd_name"] = cmds["order_name"].fillna(cmds["type_name"]).astype(str).str.lower()

    # 3) 프레임 범위 계산
    start_sec, end_sec = SPA_ATK_USA_TIME_WINDOW
    start_frame = int(start_sec * FPS)
    end_frame   = int(end_sec * FPS)

    # 4) 스페인 + 시간창 + 'attack'이 들어가는 명령만 필터
    mask_player = cmds["PlayerID"] == spa_pid
    mask_time   = (cmds["Frame"] >= start_frame) & (cmds["Frame"] <= end_frame)
    mask_attack = cmds["cmd_name"].str.contains("attack")

    candidate = cmds[mask_player & mask_time & mask_attack].copy()
    if candidate.empty:
        result["spaatkusa"] = 0
        return result

    # 5) 좌표 추출
    def extract_x(pos):
        if isinstance(pos, dict) and "X" in pos:
            return pos["X"]
        return np.nan

    def extract_y(pos):
        if isinstance(pos, dict) and "Y" in pos:
            return pos["Y"]
        return np.nan

    if "Pos" not in candidate.columns:
        result["spaatkusa"] = 0
        return result

    candidate["pos_x"] = candidate["Pos"].apply(extract_x)
    candidate["pos_y"] = candidate["Pos"].apply(extract_y)
    candidate = candidate.dropna(subset=["pos_x", "pos_y"])
    if candidate.empty:
        result["spaatkusa"] = 0
        return result

    # 6) 미국 본토 박스 안에 있는 공격 명령이 있는지 체크
    ux_min = SPA_ATK_USA_REGION["x_min"]
    ux_max = SPA_ATK_USA_REGION["x_max"]
    uy_min = SPA_ATK_USA_REGION["y_min"]
    uy_max = SPA_ATK_USA_REGION["y_max"]

    in_usa = (
        (candidate["pos_x"] >= ux_min) & (candidate["pos_x"] <= ux_max) &
        (candidate["pos_y"] >= uy_min) & (candidate["pos_y"] <= uy_max)
    )

    count_in_box = int(in_usa.sum())
    result["spaatkusa"] = 1 if count_in_box > 0 else 0

    return result


# ==================== 업그레이드 요약 ====================

def summarize_upgrades(players_df, cmds_df):
    """
    국가별 어떤 업그레이드를 몇 번 했는지 집계해서 DataFrame으로 반환.
    다만 짧은 프레임 윈도우 내에서 스팸 클릭한 경우
    한 윈도우 당 최대 UPGRADE_SPAM_CAP_PER_WINDOW까지만 인정.

    반환 컬럼:
      nation, upgrade_name, count  (스팸 처리 후 카운트)
    """
    if cmds_df.empty:
        return pd.DataFrame(columns=["nation", "upgrade_name", "count"])

    # PlayerID -> nation 매핑
    pid_to_nation = {}
    for _, row in players_df.iterrows():
        slot = row.get("SlotID")
        pid  = row.get("ID")
        nation = SLOT_TO_NATION.get(slot)
        if nation is not None and pid is not None:
            pid_to_nation[pid] = nation

    if "Upgrade" not in cmds_df.columns:
        return pd.DataFrame(columns=["nation", "upgrade_name", "count"])

    # Upgrade 명령만 필터
    up_cmds = cmds_df[cmds_df["Upgrade"].notna()].copy()
    if up_cmds.empty:
        return pd.DataFrame(columns=["nation", "upgrade_name", "count"])

    up_cmds["nation"] = up_cmds["PlayerID"].map(pid_to_nation)
    up_cmds = up_cmds[up_cmds["nation"].notna()].copy()
    if up_cmds.empty:
        return pd.DataFrame(columns=["nation", "upgrade_name", "count"])

    def get_upgrade_name(u):
        if isinstance(u, dict):
            return u.get("Name")
        return str(u)

    up_cmds["upgrade_name"] = up_cmds["Upgrade"].apply(get_upgrade_name)

    # ★ 핵심: 같은 (nation, upgrade_name)에 대해 Frame 리스트를 모아서
    #         capped_click_count_by_window로 스팸 처리
    grouped = (
        up_cmds
        .groupby(["nation", "upgrade_name"])["Frame"]
        .apply(list)
        .reset_index()
    )

    grouped["count"] = grouped["Frame"].apply(
        lambda frames: capped_click_count_by_window(
            frames,
            window_sec=UPGRADE_SPAM_WINDOW_SEC,
            cap_per_window=UPGRADE_SPAM_CAP_PER_WINDOW,
            fps=FPS
        )
    )

    # 원래대로 nation, upgrade_name, count만 사용
    grouped = (
        grouped[["nation", "upgrade_name", "count"]]
        .sort_values(["nation", "count"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return grouped
#업그레이드 스팸 처리
def capped_click_count_by_window(frames,
                                 window_sec=1.0,
                                 cap_per_window=1,
                                 fps=FPS) -> int:
    """
    같은 업그레이드에 대한 클릭 프레임 리스트를 받아서,
    '업그레이드 1번에 1초(24프레임)가 필요하다'는 가정 하에

    - 1초(= window_sec * fps) 안에 발생한 스팸 클릭은 전부 1번 시도로만 인정.
    - 그 다음 클릭이 이전 시도에서 window_frames 이상 떨어져 있으면 새로운 시도로 카운트.

    예:
      frames = [1000, 1005, 1020, 3000]
      window_sec = 1.0, fps = 24 → window_frames = 24
      1000,1005,1020 → 1초(24프레임) 이내 한 뭉치 → 1회
      3000 → 이전(1000)에서 2000프레임 떨어져 있으니 새 시도 → 1회
      총 2회로 계산.
    """
    if not frames:
        return 0

    frames_sorted = sorted(frames)
    window_frames = int(window_sec * fps)

    # 첫 클릭은 무조건 1번 시도로 인정
    total = 1
    current_start = frames_sorted[0]

    for f in frames_sorted[1:]:
        # 이전 시도 시작 프레임(current_start)로부터 window_frames 이상 떨어져 있으면
        # 새 업그레이드 시도로 본다.
        if f - current_start >= window_frames:
            total += 1
            current_start = f
        # else: 같은 1초 창 안의 스팸 → 무시 (이미 1회로 카운트했으므로)

    return total

#업그레이드 취합
def compute_up_atk_def_from_summary(up_df):
    """
    summarize_upgrades 결과를 받아
    nation별로

      - 전체 공격 업 횟수: {nation}_up_atk
      - 전체 방어 업 횟수: {nation}_up_def
      - 테란 보병 공업:    {nation}_up_tinf
      - 프로토스 지상 공업:{nation}_up_pgw
      - 저그 근접 공업:    {nation}_up_zmelee
      - 저그 원거리 공업:  {nation}_up_zmissile

    를 dict로 반환한다.
    """
    result = {}

    # 아무 업그레이드가 없으면 0으로 채운 dict 반환
    if up_df is None or up_df.empty:
        for nation in NATIONS:
            result[f"{nation}_up_atk"]      = 0
            result[f"{nation}_up_def"]      = 0
            result[f"{nation}_up_tinf"]     = 0  # Terran Infantry Weapons
            result[f"{nation}_up_pgw"]      = 0  # Protoss Ground Weapons
            result[f"{nation}_up_zmelee"]   = 0  # Zerg Melee Attacks
            result[f"{nation}_up_zmissile"] = 0  # Zerg Missile Attacks
            result[f"{nation}_up_tinf_def"]  = 0
            result[f"{nation}_up_pga_def"]   = 0
            result[f"{nation}_up_zcarapace"] = 0
        return result

    df = up_df.copy()

    df["atk_contrib"] = np.where(df["upgrade_name"].isin(ATK_UPGRADES),
                                 df["count"], 0)
    df["def_contrib"] = np.where(df["upgrade_name"].isin(DEF_UPGRADES),
                                 df["count"], 0)

    # 개별 공격 업 타입별 기여도
    df["atk_tinf"]     = np.where(df["upgrade_name"] == UP_TERRAN_INFANTRY_WEAPONS,
                                  df["count"], 0)
    df["atk_pgw"]      = np.where(df["upgrade_name"] == UP_PROTOSS_GROUND_WEAPONS,
                                  df["count"], 0)
    df["atk_zmelee"]   = np.where(df["upgrade_name"] == UP_ZERG_MELEE_ATTACKS,
                                  df["count"], 0)
    df["atk_zmissile"] = np.where(df["upgrade_name"] == UP_ZERG_MISSILE_ATTACKS,
                                  df["count"], 0)

    # 개별 방어 업 타입별 기여도 (새로 추가)
    df["def_tinf"]   = np.where(df["upgrade_name"] == UP_TERRAN_INFANTRY_ARMOR,
                                df["count"], 0)   # 테란 보병 방어
    df["def_pga"]    = np.where(df["upgrade_name"] == UP_PROTOSS_GROUND_ARMOR,
                                df["count"], 0)   # 프로토스 지상 방어
    df["def_zcara"]  = np.where(df["upgrade_name"] == UP_ZERG_CARAPACE,
                                df["count"], 0)   # 저그 갑피


    # nation별 합계
    agg = df.groupby("nation")[[
        "atk_contrib",
        "def_contrib",
        "atk_tinf",
        "atk_pgw",
        "atk_zmelee",
        "atk_zmissile",
        "def_tinf",
        "def_pga",
        "def_zcara",
    ]].sum()

    for nation in NATIONS:
        if nation in agg.index:
            row = agg.loc[nation]
            total_atk = int(row["atk_contrib"])
            total_def = int(row["def_contrib"])

            tinf     = int(row["atk_tinf"])
            pgw      = int(row["atk_pgw"])
            zmelee   = int(row["atk_zmelee"])
            zmissile = int(row["atk_zmissile"])

            tinf_def = int(row["def_tinf"])
            pga_def  = int(row["def_pga"])
            zcara    = int(row["def_zcara"])
        else:
            total_atk = total_def = 0
            tinf = pgw = zmelee = zmissile = 0
            tinf_def = pga_def = zcara = 0

        # 기존 컬럼 유지
        result[f"{nation}_up_atk"] = total_atk
        result[f"{nation}_up_def"] = total_def

        # 공격 업 세부
        result[f"{nation}_up_tinf"]     = tinf
        result[f"{nation}_up_pgw"]      = pgw
        result[f"{nation}_up_zmelee"]   = zmelee
        result[f"{nation}_up_zmissile"] = zmissile

        # 방어 업 세부 (새 컬럼)
        result[f"{nation}_up_tinf_def"]  = tinf_def       # 테란 보병 방어
        result[f"{nation}_up_pga_def"]   = pga_def        # 프로토스 지상 방어
        result[f"{nation}_up_zcarapace"] = zcara          # 저그 갑피


    return result
#====포병 기병 생산=====
def summarize_early_production(players_df,
                               cmds_df,
                               start_sec=0,
                               end_sec=300,
                               fps=FPS):
    """
    0초 ~ end_sec(기본 300초) 사이의 Train 명령을 조사해서,
    국가별로 기병(Firebat), 포병(Ghost) 생산 횟수를 세어 dict로 반환.

    반환 예:
      {
        "gbr_cav_prod": 3,
        "gbr_art_prod": 1,
        "fra_cav_prod": 0,
        "fra_art_prod": 2,
        ...
      }
    """
    result = {}
    # 기본값 0으로 초기화
    for nation in NATIONS:
        result[f"{nation}_inf_prod"] = 0  # Marine
        result[f"{nation}_cav_prod"] = 0  # Firebat
        result[f"{nation}_art_prod"] = 0  # Ghost

    if cmds_df is None or cmds_df.empty or players_df is None or players_df.empty:
        return result

    # PlayerID -> nation 매핑
    pid_to_nation = {}
    for _, row in players_df.iterrows():
        slot = row.get("SlotID")
        pid  = row.get("ID")
        nation = SLOT_TO_NATION.get(slot)
        if nation is not None and pid is not None:
            pid_to_nation[pid] = nation

    # 필요한 컬럼이 없으면 바로 반환
    if "Type" not in cmds_df.columns or "Unit" not in cmds_df.columns:
        return result

    # 시간창 → 프레임
    start_frame = int(start_sec * fps)
    end_frame   = int(end_sec * fps)

    cmds = cmds_df[
        (cmds_df["Frame"] >= start_frame) &
        (cmds_df["Frame"] <= end_frame)
    ].copy()
    if cmds.empty:
        return result

    # Type.Name, Unit.Name 추출
    def get_type_name(t):
        if isinstance(t, dict) and "Name" in t:
            return t["Name"]
        return t

    def get_unit_name(u):
        if isinstance(u, dict) and "Name" in u:
            return u["Name"]
        return u

    cmds["type_name"] = cmds["Type"].apply(get_type_name).astype(str)
    cmds["unit_name"] = cmds["Unit"].apply(get_unit_name).astype(str)

    # Train 명령만 필터
    train_cmds = cmds[cmds["type_name"].str.lower() == "train"].copy()
    if train_cmds.empty:
        return result

    # nation 붙이기
    train_cmds["nation"] = train_cmds["PlayerID"].map(pid_to_nation)
    train_cmds = train_cmds[train_cmds["nation"].notna()].copy()
    if train_cmds.empty:
        return result

    # 유닛 타입별 마스크
    inf_mask = train_cmds["unit_name"] == "Marine"   # 보병
    cav_mask = train_cmds["unit_name"] == "Firebat"  # 기병
    art_mask = train_cmds["unit_name"] == "Ghost"    # 포병

    # nation별 개수 집계
    inf_counts = train_cmds[inf_mask].groupby("nation").size()
    cav_counts = train_cmds[cav_mask].groupby("nation").size()
    art_counts = train_cmds[art_mask].groupby("nation").size()
    
    for nation, cnt in inf_counts.items():
       result[f"{nation}_inf_prod"] = int(cnt)
    for nation, cnt in cav_counts.items():
        result[f"{nation}_cav_prod"] = int(cnt)
    for nation, cnt in art_counts.items():
        result[f"{nation}_art_prod"] = int(cnt)

    return result
#턴 보병 생산
def summarize_infantry_by_turn(players_df,
                               cmds_df,
                               turn_len_sec=300,
                               max_turns=3,
                               fps=FPS):
    """
    turn_len_sec(기본 300초) 단위로 0~turn_len, turn_len~2*turn_len, ...
    최대 max_turns까지 잘라서,
    각 구간마다 Marine(보병) 생산 횟수를 국가별로 세어 dict로 반환.

    반환 예:
      {
        "gbr_inf_t1": 5,
        "gbr_inf_t2": 3,
        ...
        "tur_inf_t10": 0,
      }
    """
    result = {}

    # 우선 0으로 초기화
    for nation in NATIONS:
        for t in range(1, max_turns + 1):
            result[f"{nation}_inf_t{t}"] = 0

    if cmds_df is None or cmds_df.empty:
        return result

    # PlayerID -> nation 매핑
    pid_to_nation = {}
    for _, row in players_df.iterrows():
        slot = row.get("SlotID")
        pid  = row.get("ID")
        nation = SLOT_TO_NATION.get(slot)
        if nation is not None and pid is not None:
            pid_to_nation[pid] = nation

    # 필요한 컬럼 없으면 패스
    if "Type" not in cmds_df.columns or "Unit" not in cmds_df.columns or "Frame" not in cmds_df.columns:
        return result

    # Type.Name / Unit.Name 정규화
    def get_type_name(t):
        if isinstance(t, dict) and "Name" in t:
            return t["Name"]
        return t

    def get_unit_name(u):
        if isinstance(u, dict) and "Name" in u:
            return u["Name"]
        return u

    cmds = cmds_df.copy()
    cmds["type_name"] = cmds["Type"].apply(get_type_name).astype(str)
    cmds["unit_name"] = cmds["Unit"].apply(get_unit_name).astype(str)

    # Train 명령만
    train_cmds = cmds[cmds["type_name"].str.lower() == "train"].copy()
    if train_cmds.empty:
        return result

    # nation 붙이기
    train_cmds["nation"] = train_cmds["PlayerID"].map(pid_to_nation)
    train_cmds = train_cmds[train_cmds["nation"].notna()].copy()
    if train_cmds.empty:
        return result

    # 턴별로 잘라서 Marine만 세기
    for t in range(max_turns):
        start_sec  = t * turn_len_sec
        end_sec    = (t + 1) * turn_len_sec
        start_fr   = int(start_sec * fps)
        end_fr     = int(end_sec * fps)

        window = train_cmds[
            (train_cmds["Frame"] >= start_fr) &
            (train_cmds["Frame"] <  end_fr)
        ]
        if window.empty:
            continue

        inf_window = window[window["unit_name"] == "Marine"]
        if inf_window.empty:
            continue

        counts = inf_window.groupby("nation").size()
        for nation, cnt in counts.items():
            result[f"{nation}_inf_t{t+1}"] = int(cnt)

    return result

# ==================== 승패(iswin / isrevwin) 추론 ====================

def infer_iswin(players_df, chat_df, leave_df, fps=FPS):
    """
    ChatCmds + LeaveGameCmds로 팀 승패 추론.
    - 3분 이후 첫 LeaveGame 또는 가장 먼저 'ㅈㅈ' 친 사람의 팀을 패배팀으로 본다.
    """
    # ID/Slot → nation
    id_to_nation = {}
    slot_to_nation = {}
    for _, row in players_df.iterrows():
        slot = row.get("SlotID")
        pid  = row.get("ID")
        nation = SLOT_TO_NATION.get(slot)
        if nation is not None:
            slot_to_nation[slot] = nation
        if nation is not None and pid is not None:
            id_to_nation[pid] = nation

    result = {}
    for nation in SLOT_TO_NATION.values():
        result[f"{nation}_iswin"] = np.nan
    result["isrevwin"]     = np.nan
    result["loser_nation"] = np.nan
    result["loser_team"]   = np.nan

    if not id_to_nation and not slot_to_nation:
        return result

    # 1) 'ㅈㅈ' 채팅
    first_gg_row = None
    if not chat_df.empty and "Message" in chat_df.columns:
        def has_gg(msg):
            if not isinstance(msg, str):
                return False
            return any(pat in msg for pat in GG_PATTERNS)
        gg_rows = chat_df[chat_df["Message"].apply(has_gg)].copy()
        if not gg_rows.empty:
            first_gg_row = gg_rows.sort_values("Frame").iloc[0]

    # 2) 3분 이후 첫 LeaveGame
    first_leave_row = None
    if not leave_df.empty:
        min_frame_allowed = EARLY_LEAVE_SEC * fps
        leaves_late = leave_df[leave_df["Frame"] >= min_frame_allowed].copy()
        if not leaves_late.empty:
            first_leave_row = leaves_late.sort_values("Frame").iloc[0]

    # 3) 더 먼저 일어난 쪽 선택
    loser_row = None
    loser_source = None  # "gg" or "leave"

    if first_gg_row is not None and first_leave_row is not None:
        if first_gg_row["Frame"] <= first_leave_row["Frame"]:
            loser_row = first_gg_row
            loser_source = "gg"
        else:
            loser_row = first_leave_row
            loser_source = "leave"
    elif first_gg_row is not None:
        loser_row = first_gg_row
        loser_source = "gg"
    elif first_leave_row is not None:
        loser_row = first_leave_row
        loser_source = "leave"

    if loser_row is None:
        return result

    # 4) 패배 플레이어의 nation 찾기
    loser_nation = None
    if loser_source == "gg":
        slot = loser_row.get("SenderSlotID")
        loser_nation = slot_to_nation.get(slot)
    elif loser_source == "leave":
        pid = loser_row.get("PlayerID")
        loser_nation = id_to_nation.get(pid)

    if loser_nation is None:
        return result

    loser_team = NATION_TO_TEAM.get(loser_nation)
    if loser_team is None:
        return result

    # 5) 팀별 iswin
    for nation in SLOT_TO_NATION.values():
        if nation == "com":
            result[f"{nation}_iswin"] = np.nan
            continue
        team = NATION_TO_TEAM.get(nation)
        if team is None:
            result[f"{nation}_iswin"] = np.nan
        else:
            result[f"{nation}_iswin"] = 0 if team == loser_team else 1

    # 6) 혁명 승리 여부
    if loser_team == "revolt":
        result["isrevwin"] = 0
    elif loser_team == "allies":
        result["isrevwin"] = 1
    else:
        result["isrevwin"] = np.nan

    result["loser_nation"] = loser_nation
    result["loser_team"]   = loser_team

    return result


# ==================== 기타 유틸: duration 변환 ====================

def frames_to_duration(frames, fps=FPS):
    if frames is None:
        return np.nan
    try:
        if np.isnan(frames):
            return np.nan
    except TypeError:
        pass
    return frames / fps


# ==================== 리플레이 → 한 줄 요약 ====================

def replay_to_summary_row(rep_path):
    """
    한 리플레이에서:
      - 각국 name, eapm, trait, focus
      - 이벤트: sweden, balkan, denmark, canada, warsaw
      - 업그레이드: nation_up_atk, nation_up_def
      - 승패: 각국 iswin, isrevwin
      - frames, duration, title
    모두 담은 Series 반환.
    """
    header, players_df, cmds_df, pdescs_df, chat_df, leave_df = load_replay(rep_path)
    merged = merge_players_with_descriptors(players_df, pdescs_df)
    eapm_col = get_eapm_col(pdescs_df)

    result = {}

    # EAPM + 이름
    for slot_id, nation in SLOT_TO_NATION.items():
        row = merged.loc[merged["SlotID"] == slot_id]
        if len(row) == 0:
            result[f"{nation}_eapm"] = np.nan
            result[f"{nation}_name"] = np.nan
        else:
            r = row.iloc[0]
            name_value = r.get("Name", "")
            eapm_value = r[eapm_col] if eapm_col is not None else np.nan
            result[f"{nation}_eapm"] = eapm_value
            result[f"{nation}_name"] = name_value

    # trait / focus / 이벤트
    result.update(infer_all_traits(players_df, cmds_df))
    result.update(infer_all_focuses(players_df, cmds_df))
    result.update(infer_events(players_df, cmds_df))
    
    # 영국 드랍쉽 상륙 여부
    result.update(detect_gbr_dropship_landing(players_df, cmds_df))

    # 영국 첫 턴 공격 위치(미국/오스만) 여부
    result.update(detect_gbr_first_turn_attack(players_df, cmds_df))

    # 프랑스의 영국 본토 침공 여부
    result.update(detect_fra_invasion(players_df, cmds_df))
    
    #프랑 러시아 공격
    result.update(fraatkrus(players_df, cmds_df))
    #오스만 이집트
    result.update(turatkegypt(players_df, cmds_df))
    
    # 업그레이드 요약
    up_summary = summarize_upgrades(players_df, cmds_df)
    result.update(compute_up_atk_def_from_summary(up_summary))
    
    # 초반(0~300초) 기병/포병 생산 요약
    prod_summary = summarize_early_production(players_df, cmds_df,
                                          start_sec=0,
                                          end_sec=300,
                                          fps=FPS)
    result.update(prod_summary)
    # 턴별 보병(Marine) 생산 요약
    inf_turn_summary = summarize_infantry_by_turn(
       players_df,
       cmds_df,
       turn_len_sec=300,  # 1턴 길이(초)
       max_turns=3,
       fps=FPS,
    )
    result.update(inf_turn_summary)
    #유동 첫턴 전투 기간
    hre_fra_battle = infer_HREFRAbattledur(players_df, cmds_df)
    result.update(hre_fra_battle)
    #미국 캐나다 침공
    result.update(usaatkcan(players_df, cmds_df))
    #스페인 미국 본토 공격
    result.update(spaatkusa(players_df, cmds_df))
    #터키
    result.update(turatkind(players_df, cmds_df))
    #프랑남미
    result.update(fraatksam(players_df, cmds_df))
    #러시아 중동
    result.update(detect_rus_invasion_tur(players_df, cmds_df)) 
    #영국이탈리아상륙
    result.update(detect_gbr_invasion_italy(players_df, cmds_df))
    #미국멕시코
    result.update(usaatkmex(players_df, cmds_df))
    #이베리아
    result.update(fraatkpor(players_df, cmds_df))
    result.update(gbrdefspa(players_df, cmds_df))
    #동맹국 프랑스 본토 침공
    result.update(detect_allies_invasion_fra(players_df, cmds_df))
    # 이벤트
    events = infer_events(players_df, cmds_df)
    result.update(events)
    
    # 덴마크/노르웨이 전역 러시아 공격/점령 변수
    den_val = events.get("denmark", np.nan)
    rus_den_vars = infer_rus_den_nor(players_df, cmds_df, den_val)
    result.update(rus_den_vars)

    # 승패
    result.update(infer_iswin(players_df, chat_df, leave_df))

    # 메타
    frames = header.get("Frames", np.nan)
    result["frames"]   = frames
    result["duration"] = frames_to_duration(frames)
    result["title"]    = header.get("Title", "")

    return pd.Series(result)


# ==================== 폴더 단위 처리 + 엑셀 저장 ====================

def replays_folder_to_df(folder):
    rows = []
    pattern = os.path.join(folder, "*.rep")
    for rep_path in glob.glob(pattern):
        try:
            row = replay_to_summary_row(rep_path)
            row["replay_file"] = os.path.basename(rep_path)
            rows.append(row)
        except Exception as e:
            print(f"에러 발생: {rep_path} -> {e}")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def replays_to_excel(folder, out_path):
    """
    folder 안의 리플레이들을 모두 분석해서,
    엑셀 파일(out_path)에 저장.

    컬럼 구조(가능하면 이 순서로 정렬):
      gameid, duration
      { nation_name, nation_eapm, nation_trait, nation_focus, nation_up_atk, nation_up_def } * 7국가
      warsaw, canada, sweden, balkan, denmark
      gbr_dropship_landing, gbratkusa, gbratktur
      is_a_win, is_r_win
    """
    df = replays_folder_to_df(folder)
    if df.empty:
        print("리플레이 분석 결과가 비었습니다.")
        return df

    # gameid 생성
    if "replay_file" in df.columns:
        df["gameid"] = df["replay_file"].str.replace(".rep", "", regex=False)
    else:
        df["gameid"] = np.arange(len(df))

    # is_r_win / is_a_win 생성 (infer_iswin 결과에서 isrevwin 사용)
    if "isrevwin" in df.columns:
        df["is_r_win"] = df["isrevwin"]

        def compute_is_a_win(v):
            if pd.isna(v):
                return np.nan
            return 1 - v

        df["is_a_win"] = df["is_r_win"].apply(compute_is_a_win)
    else:
        df["is_r_win"] = np.nan
        df["is_a_win"] = np.nan

    # 새로 만든 파생변수(영국 드랍 + 첫 턴 공격) 컬럼이 없으면 만들어 두기
    # (예전 버전 코드로 만든 df에도 에러 없이 동작하도록)
    extra_cols = ["gbr_dropship_landing", "gbratkusa", "gbratktur", "gbrdefspa",
                  "gbrinvita", "gbr_moveunload_cnt_italy",
                  "fra_invasion", "fraatksam", "fraatksam_cmd_count", "fraatkpor",
                  "fraatkrus", "fraatkrus_cmd_count", "isparisatked", "isparisatked_cmd_count",
                  "HREFRA_battledur", "HREFRA_firstbattle",
                  "rusatkden", "usadefden", "rusconqnor", "norwardur",
                  "rusfastinv", "rusnorinv", "spaatkusa",
                  "usaatkcan_count", "usaatkcan","usaatkmex", "usaatkmex_cmd_count",
                  "turatkindia", "tur_moveunload_cnt_india", "tur_moveatk_cnt_india",
                  "turatkegypt","turatkegypt_cmd_count"]
    for col in extra_cols:
        if col not in df.columns:
            df[col] = np.nan

    # ===== 컬럼 순서 구성 =====
    col_order = ["gameid", "duration"]

    # 국가별 기본 정보
    for nation in NATIONS:
        col_order.append(f"{nation}_name")
        col_order.append(f"{nation}_eapm")
        col_order.append(f"{nation}_trait")
        col_order.append(f"{nation}_focus")
        col_order.append(f"{nation}_up_tinf")
        col_order.append(f"{nation}_up_pgw")
        col_order.append(f"{nation}_up_zmelee")
        col_order.append(f"{nation}_up_zmissile")
        col_order.append(f"{nation}_up_tinf_def")
        col_order.append(f"{nation}_up_pga_def")
        col_order.append(f"{nation}_up_zcarapace")
        for t in range(1, 4):  # max_turns와 맞추기
           col_order.append(f"{nation}_inf_t{t}")
        col_order.append(f"{nation}_cav_prod")
        col_order.append(f"{nation}_art_prod")

    
    # (맵) 컬럼
    base_events = ["warsaw", "canada", "sweden", "balkan", "denmark"]

    # 원본 0/1/2 변수는 엑셀에 안 쓰고,
    # is_XXX_on / XXX_decision 만 포함
    event_flag_cols = [f"is_{ev}_on" for ev in base_events]
    event_dec_cols  = [f"{ev}_decision" for ev in base_events]

    col_order.extend(event_flag_cols + event_dec_cols)

    # 영국 드랍/첫 턴 공격 파생변수
    col_order.extend(extra_cols)

    # 승패 컬럼
    col_order.extend(["is_a_win", "is_r_win"])

    # 실제 df에 존재하는 컬럼만 사용
    final_cols = [c for c in col_order if c in df.columns]
    df_out = df[final_cols].copy()

    df_out.to_excel(out_path, index=False)
    print(f"엑셀 저장 완료: {out_path}")
    return df_out



# ==================== 직접 실행 예시 ====================

if __name__ == "__main__":

    # 폴더 전체 → 엑셀 저장
    folder = r"C:\Users\korrl\OneDrive\npw\replay\1.7.9.1"
    out_xlsx = r"C:\Users\korrl\OneDrive\npw\프본.xlsx"
    df_result = replays_to_excel(folder, out_xlsx)
    print(df_result.head())
