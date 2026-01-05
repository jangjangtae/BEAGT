import numpy as np
import gymnasium as gym
from gymnasium import spaces
import MalmoPython
import time
import json
import math

class SimpleVoxelEnv(gym.Env):
    def __init__(self, port=10006): # [체크] 포트 번호 확인
        super().__init__()
        self.grid_shape = (5, 3, 5) 
        self.obs_dim = 5 * 3 * 5
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.obs_dim,), dtype=np.float32)
        
        # 걷기만 가능 (점프 X)
        self.action_list = ["move 1", "turn 1", "turn -1", "move 0"] 
        self.action_space = spaces.Discrete(len(self.action_list))
        
        self.port = port
        self.agent_host = MalmoPython.AgentHost()
        self.client_pool = MalmoPython.ClientPool()
        self.client_pool.add(MalmoPython.ClientInfo("127.0.0.1", self.port))
        
        self.visited_cells = set()
        self.block_map = {"air": 0, "stone": 1, "bedrock": 2, "gold_block": 3, "diamond_block": 4, "glass": 5, "glowstone": 6}
        
        # 맵 설정 (50층)
        self.start_height = 50 
        self.goal_height = 2   
        
        self.spawn_x = 0
        self.spawn_z = 0

    def _get_mission_xml(self):
        draw_cmds = ""
        # 1. 맵 전체 초기화
        draw_cmds += '<DrawCuboid x1="-15" y1="1" z1="-15" x2="15" y2="60" z2="15" type="air"/>'
        # 2. 바닥 (Bedrock)
        draw_cmds += '<DrawCuboid x1="-15" y1="1" z1="-15" x2="15" y2="1" z2="15" type="bedrock"/>'

        # 3. [업그레이드] 더 거대한 유리 요새 (Radius 4 -> 9x9)
        # 내부 공간을 더 넉넉하게 감싸서 절대 밖으로 못 나가게 함
        draw_cmds += f'<DrawCuboid x1="-4" y1="1" z1="-4" x2="4" y2="{self.start_height+5}" z2="4" type="glass"/>'
        
        # 중심 기둥 (조명)
        draw_cmds += f'<DrawCuboid x1="0" y1="1" z1="0" x2="0" y2="{self.start_height+5}" z2="0" type="glowstone"/>'
        
        # 4. 경로 좌표 생성 (5x5 Ring)
        path_coords = []
        # (1) North Wall (x: -2 -> 2, z: 2)
        for x in range(-2, 3): path_coords.append((x, 2))
        # (2) East Wall (x: 2, z: 2 -> -2)
        for z in range(1, -3, -1): path_coords.append((2, z))
        # (3) South Wall (x: 2 -> -2, z: -2)
        for x in range(1, -3, -1): path_coords.append((x, -2))
        # (4) West Wall (x: -2, z: -2 -> 1) 
        for z in range(-1, 2): path_coords.append((-2, z))
        
        total_steps = len(path_coords)
        current_y = self.start_height
        idx = 0
        
        # 시작점 설정
        self.spawn_x = path_coords[0][0] # (0, 2)
        self.spawn_z = path_coords[0][1]
        
        # [핵심] 3x3 안전 발판 생성 (Safety Platform)
        # 스폰 지점을 중심으로 3x3 영역을 돌로 채움
        # 유리 기둥이 R=4이므로 R=1인 3x3 플랫폼은 안전하게 들어감
        draw_cmds += f'<DrawCuboid x1="{self.spawn_x-1}" y1="{self.start_height-1}" z1="{self.spawn_z-1}" x2="{self.spawn_x+1}" y2="{self.start_height-1}" z2="{self.spawn_z+1}" type="stone"/>'
        
        # 머리 위 공간 확보 (3x3)
        draw_cmds += f'<DrawCuboid x1="{self.spawn_x-1}" y1="{self.start_height}" z1="{self.spawn_z-1}" x2="{self.spawn_x+1}" y2="{self.start_height+3}" z2="{self.spawn_z+1}" type="air"/>'
        
        # 추가 조명
        draw_cmds += f'<DrawBlock x="{self.spawn_x}" y="{self.start_height+3}" z="{self.spawn_z}" type="glowstone"/>'

        # 나선형 하강
        while current_y > self.goal_height:
            x, z = path_coords[idx % total_steps]
            nx, nz = path_coords[(idx + 1) % total_steps]
            
            # (1) 현재 위치 파내기 (터널)
            draw_cmds += f'<DrawCuboid x1="{x}" y1="{current_y}" z1="{z}" x2="{x}" y2="{current_y+2}" z2="{z}" type="air"/>'
            
            # (2) 바닥 깔기 (금 블록)
            # 단, 시작 지점 근처(idx < 4)는 이미 돌 발판이 있으므로 덮어쓰지 않거나 덮어써도 무방
            draw_cmds += f'<DrawBlock x="{x}" y="{current_y-1}" z="{z}" type="gold_block"/>'
            
            # (3) 다음 칸 미리 뚫기
            draw_cmds += f'<DrawCuboid x1="{nx}" y1="{current_y}" z1="{nz}" x2="{nx}" y2="{current_y+2}" z2="{nz}" type="air"/>'
            
            idx += 1
            
            # [안전] 처음 4걸음은 평지로 유지 (Start Flat)
            # 바로 내려가지 않고 조금 걷게 해서 적응 유도
            if idx > 4:
                current_y -= 1
            
        # 5. 도착 지점
        lx, lz = path_coords[idx % total_steps]
        draw_cmds += f'<DrawCuboid x1="{lx}" y1="{self.goal_height}" z1="{lz}" x2="{lx}" y2="{self.goal_height+2}" z2="{lz}" type="air"/>'
        draw_cmds += f'<DrawBlock x="{lx}" y="{self.goal_height-1}" z="{lz}" type="diamond_block"/>'

        return f'''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
        <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
          <About><Summary>Sealed Safe Spawn V2</Summary></About>
          <ServerSection>
            <ServerHandlers>
              <FlatWorldGenerator generatorString="3;7,2;1;"/>
              <DrawingDecorator>{draw_cmds}</DrawingDecorator>
              <ServerQuitWhenAnyAgentFinishes/>
            </ServerHandlers>
          </ServerSection>
          <AgentSection mode="Survival">
            <Name>VoxelAgent</Name>
            <AgentStart><Placement x="{self.spawn_x}.5" y="{self.start_height}" z="{self.spawn_z}.5" yaw="0"/></AgentStart>
            <AgentHandlers>
              <ObservationFromGrid>
                <Grid name="surrounding_blocks">
                  <min x="-2" y="-1" z="-2"/>
                  <max x="2"  y="1"  z="2"/>
                </Grid>
              </ObservationFromGrid>
              <ObservationFromFullStats/>
              <ContinuousMovementCommands/>
              <AbsoluteMovementCommands/>
            </AgentHandlers>
          </AgentSection>
        </Mission>'''

    def step(self, action_idx):
        action = self.action_list[action_idx]
        try: self.agent_host.sendCommand(action)
        except RuntimeError: pass
        time.sleep(0.02)
        
        ws = self.agent_host.getWorldState()
        obs = self._get_observation(ws)
        
        reward = 0.0
        done = False
        info = {}

        # [안전장치]
        x, y, z = 0, self.start_height, 0 
        has_valid_data = False

        if ws.number_of_observations_since_last_state > 0:
            try:
                msg = ws.observations[-1].text
                data = json.loads(msg)
                
                if 'XPos' in data and 'YPos' in data:
                    x = data.get(u'XPos')
                    y = data.get(u'YPos')
                    z = data.get(u'ZPos')
                    has_valid_data = True
                
                info['XPos'] = x
                info['YPos'] = y
                info['ZPos'] = z
                
                cell = (int(x), int(y), int(z))
                if cell not in self.visited_cells:
                    self.visited_cells.add(cell)
            except:
                pass

        if has_valid_data:
            if y <= self.goal_height + 0.5:
                reward = 100.0
                done = True
                print(f"🎉 [Success] Reached Floor! (Height: {y:.2f})")
                
        info["visited_count"] = len(self.visited_cells)
        
        if not ws.is_mission_running:
            done = True
            
        return obs, reward, done, False, info

    def _get_observation(self, ws):
        grid_vec = np.zeros(self.obs_dim, dtype=np.float32)
        if ws.number_of_observations_since_last_state > 0:
            try:
                msg = ws.observations[-1].text
                data = json.loads(msg)
                if "surrounding_blocks" in data:
                    grid = data["surrounding_blocks"]
                    grid_vec = np.array([self.block_map.get(b, 0) for b in grid], dtype=np.float32)
            except: pass
        return grid_vec

    def reset(self, seed=None, options=None):
        if self.spawn_x == 0:
            self._get_mission_xml()
            
        try:
            self.agent_host.sendCommand(f"tp {self.spawn_x}.5 {self.start_height} {self.spawn_z}.5")
            self.agent_host.sendCommand("setYaw 0")
        except: pass

        world_state = self.agent_host.getWorldState()
        if not world_state.is_mission_running:
            my_mission = MalmoPython.MissionSpec(self._get_mission_xml(), True)
            try:
                self.agent_host.startMission(my_mission, self.client_pool, MalmoPython.MissionRecordSpec(), 0, "voxel_exp")
                while not self.agent_host.getWorldState().has_mission_begun:
                    time.sleep(0.1)
            except:
                time.sleep(1)

        print("Waiting for spawn...", end="")
        timeout = 0
        while True:
            ws = self.agent_host.getWorldState()
            if ws.number_of_observations_since_last_state > 0:
                msg = ws.observations[-1].text
                data = json.loads(msg)
                if 'YPos' in data and data['YPos'] > 10:
                    print(f" Ready at Y={data['YPos']:.1f}!")
                    break
            time.sleep(0.1)
            timeout += 1
            if timeout > 50:
                self.agent_host.sendCommand(f"tp {self.spawn_x}.5 {self.start_height} {self.spawn_z}.5")
                timeout = 0

        self.visited_cells.clear()
        return self._get_observation(ws), {}
    
    def close(self): pass
