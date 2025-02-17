# visual_simulation.py

import pygame
import random
from Environment import WardEnv  # Import your existing environment code

# -------------------------------
# Initialize Pygame and Settings
# -------------------------------
pygame.init()

# Define panel dimensions
GRID_AREA_WIDTH = 600   # Left area: simulation grid
RESOURCE_PANEL_WIDTH = 200  # Right area: resource dashboard
WINDOW_WIDTH = GRID_AREA_WIDTH + RESOURCE_PANEL_WIDTH  # Total width: 800px
WINDOW_HEIGHT = 600

# Grid settings for the simulation area
GRID_ROWS, GRID_COLS = 20, 20  # The grid is 20x20 cells in the simulation area
CELL_WIDTH = GRID_AREA_WIDTH // GRID_COLS  # e.g., 600 // 20 = 30 pixels per cell
CELL_HEIGHT = WINDOW_HEIGHT // GRID_ROWS     # 600 // 20 = 30 pixels per cell

FPS = 5  # Slow FPS for clarity

# -------------------------------
# Define Colors
# -------------------------------
BLACK   = (0, 0, 0)
WHITE   = (255, 255, 255)
BLUE    = (50, 50, 255)
GREEN   = (50, 255, 50)
RED     = (255, 50, 50)
YELLOW  = (255, 255, 50)
GRAY    = (200, 200, 200)
CORRIDOR_COLOR = (100, 100, 100)  # Color for pathways
PANEL_BG = (30, 30, 30)  # Dark background for resource panel

# Create display
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Hospital Simulation with Resource Panel")
clock = pygame.time.Clock()

# -------------------------------
# Define Hospital Areas with Corridors
# -------------------------------
# Updated positions so that wards are distributed across the grid.
areas = [
    {   # Ward A positioned in the upper left
        "name": "Ward A",
        "type": "ward",
        "start_row": 1,
        "start_col": 1,
        "rows": 5,
        "cols": 5,
        "color": BLUE
    },
    {   # Ward B positioned more centrally/right
        "name": "Ward B",
        "type": "ward",
        "start_row": 1,
        "start_col": 10,
        "rows": 5,
        "cols": 5,
        "color": GREEN
    },
    {   # Waiting Room positioned in the lower center
        "name": "Waiting Room",
        "type": "waiting",
        "start_row": 10,
        "start_col": 5,
        "rows": 5,
        "cols": 5,
        "color": GRAY
    }
]

# -------------------------------
# Helper Functions
# -------------------------------
def is_in_area(row, col, area):
    """Return True if the grid cell (row, col) is within the given area."""
    return (row >= area["start_row"] and row < area["start_row"] + area["rows"] and
            col >= area["start_col"] and col < area["start_col"] + area["cols"])

def draw_corridors():
    """
    For each grid cell in the simulation area that is not part of a defined area,
    fill it with the corridor color.
    """
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            if not any(is_in_area(row, col, area) for area in areas):
                rect = pygame.Rect(col * CELL_WIDTH, row * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT)
                pygame.draw.rect(screen, CORRIDOR_COLOR, rect)

def draw_grid():
    """Draws the underlying grid for visual reference in the simulation area."""
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            rect = pygame.Rect(col * CELL_WIDTH, row * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT)
            pygame.draw.rect(screen, WHITE, rect, 1)

def draw_hospital_layout():
    """
    Draws the wards and waiting room with thicker borders and centered labels.
    These areas are drawn on the simulation grid.
    """
    font = pygame.font.SysFont("Arial", 16)
    for area in areas:
        x = area["start_col"] * CELL_WIDTH
        y = area["start_row"] * CELL_HEIGHT
        width = area["cols"] * CELL_WIDTH
        height = area["rows"] * CELL_HEIGHT
        pygame.draw.rect(screen, area["color"], (x, y, width, height), 3)
        text = font.render(area["name"], True, area["color"])
        text_rect = text.get_rect(center=(x + width // 2, y + height // 2))
        screen.blit(text, text_rect)

def draw_elements(waiting, beds, staff):
    """Overlay simulation elements (waiting patients, beds, staff) on the simulation grid."""
    for (col, row) in waiting:
        rect = pygame.Rect(col * CELL_WIDTH + 2, row * CELL_HEIGHT + 2, CELL_WIDTH - 4, CELL_HEIGHT - 4)
        pygame.draw.rect(screen, RED, rect)
    for (col, row) in beds:
        rect = pygame.Rect(col * CELL_WIDTH + 2, row * CELL_HEIGHT + 2, CELL_WIDTH - 4, CELL_HEIGHT - 4)
        pygame.draw.rect(screen, BLUE, rect)
    for (col, row) in staff:
        rect = pygame.Rect(col * CELL_WIDTH + 2, row * CELL_HEIGHT + 2, CELL_WIDTH - 4, CELL_HEIGHT - 4)
        pygame.draw.rect(screen, GREEN, rect)

def draw_resource_panel(resource_info):
    """Draws a dedicated resource panel on the right side of the window."""
    # Define panel rectangle (right side)
    panel_rect = pygame.Rect(GRID_AREA_WIDTH, 0, RESOURCE_PANEL_WIDTH, WINDOW_HEIGHT)
    pygame.draw.rect(screen, PANEL_BG, panel_rect)
    
    # Prepare font and starting positions
    font = pygame.font.SysFont("Arial", 18)
    x_start = GRID_AREA_WIDTH + 10
    y_start = 10
    line_spacing = 25

    # Display resource information
    resources = [
        f"Waiting Patients: {resource_info.get('waiting_patients', 0)}",
        f"Total Beds: {resource_info.get('total_beds', 0)}",
        f"Occupied Beds: {resource_info.get('occupied_beds', 0)}",
        f"Available Beds: {resource_info.get('available_beds', 0)}",
        f"Staff Available: {resource_info.get('staff', 0)}"
    ]
    
    for i, line in enumerate(resources):
        text_surface = font.render(line, True, YELLOW)
        screen.blit(text_surface, (x_start, y_start + i * line_spacing))

def get_resource_info(env_state, ward_config):
    """
    Compute and return a dictionary of resource metrics.
    For this example, we assume:
      - Total beds is taken from the ward configuration.
      - Occupied beds is computed using the occupancy ratio and total capacity.
      - Available beds is the remainder.
      - Staff is a fixed number (or computed if state information is available).
    """
    total_beds = ward_config.get('num_beds', 0)
    # In this layout, assume each ward can hold up to 10 beds.
    total_possible_beds = 2 * 10  # two wards
    occupied_beds = int(env_state.get('occupied_ratio', 0) * total_possible_beds)
    available_beds = total_beds - occupied_beds if total_beds >= occupied_beds else 0
    staff = env_state.get('staff_count', 0)  # This should be part of your simulation state
    return {
        "waiting_patients": env_state.get("waiting_patients", 0),
        "total_beds": total_beds,
        "occupied_beds": occupied_beds,
        "available_beds": available_beds,
        "staff": staff
    }

def map_state_to_positions(env_state):
    """
    Map the simulation state to positions within our defined areas.
    - Place waiting patients inside the waiting room.
    - Distribute beds randomly within each ward.
    - Place staff near the center of a random ward.
    """
    waiting_positions = []
    bed_positions = []
    
    waiting_area = None
    ward_areas = []
    for area in areas:
        if area["type"] == "waiting":
            waiting_area = area
        elif area["type"] == "ward":
            ward_areas.append(area)
    
    # Place waiting patients in the waiting room
    if waiting_area:
        sr = waiting_area["start_row"]
        sc = waiting_area["start_col"]
        rsize = waiting_area["rows"]
        csize = waiting_area["cols"]
        for _ in range(env_state.get('waiting_patients', 0)):
            row = random.randint(sr, sr + rsize - 1)
            col = random.randint(sc, sc + csize - 1)
            waiting_positions.append((col, row))
    
    # Place beds in each ward.
    total_possible_beds = len(ward_areas) * 10
    num_occupied = int(env_state.get('occupied_ratio', 0) * total_possible_beds)
    beds_per_area = num_occupied // len(ward_areas) if ward_areas else 0
    
    for ward in ward_areas:
        sr = ward["start_row"]
        sc = ward["start_col"]
        rsize = ward["rows"]
        csize = ward["cols"]
        for _ in range(beds_per_area):
            row = random.randint(sr, sr + rsize - 1)
            col = random.randint(sc, sc + csize - 1)
            bed_positions.append((col, row))
    
    # Place staff near the center of one random ward.
    staff_positions = []
    if ward_areas:
        chosen_ward = random.choice(ward_areas)
        center_row = chosen_ward["start_row"] + chosen_ward["rows"] // 2
        center_col = chosen_ward["start_col"] + chosen_ward["cols"] // 2
        staff_positions.append((center_col, center_row))
    else:
        staff_positions.append((GRID_COLS // 2, GRID_ROWS // 2))
    
    return waiting_positions, bed_positions, staff_positions

# -------------------------------
# Main Simulation Loop
# -------------------------------
def main():
    # Sample environment configuration
    ward_configs = {
        'general': {
            'num_beds': 40,
            'base_staff': {
                'day': {'nurses': 10, 'doctors': 5},
                'night': {'nurses': 6, 'doctors': 3}
            },
            'sim_duration': 168,
            'arrival_lambda': 0.8,
            'nurse_efficiency': 1.5,
            'treatment_params': {'shape': 3, 'scale': 20}
        }
    }
    env = WardEnv(ward_name='general', **ward_configs['general'])
    env.reset()
    
    running = True
    timestep = 0
    font_info = pygame.font.SysFont("Arial", 18)
    
    while running and timestep < env.sim_duration:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Simulate a random action (replace with your own logic)
        action = {
            'staff_adjustment': {'nurses': random.choice([-1, 0, 1]),
                                 'doctors': random.choice([-1, 0, 1])},
            'admission_priority': random.choice(['severity', 'fifo'])
        }
        next_state, reward, done = env.step(action)
        
        # Clear the simulation area (left panel) and resource panel (right)
        # Fill the entire screen with BLACK first
        screen.fill(BLACK)
        
        # Draw simulation components on the left
        draw_corridors()
        draw_grid()
        draw_hospital_layout()
        
        waiting_positions, bed_positions, staff_positions = map_state_to_positions(env.get_state())
        draw_elements(waiting_positions, bed_positions, staff_positions)
        
        # Compute resource info from the simulation state
        resource_info = get_resource_info(env.get_state(), ward_configs['general'])
        # Draw the resource panel on the right
        draw_resource_panel(resource_info)
        
        # Optionally, display simulation info (time, reward) at the top of the simulation area
        info_text = font_info.render(f"Time: {env.time} | Reward: {reward:.2f}", True, YELLOW)
        screen.blit(info_text, (10, 10))
        
        pygame.display.flip()
        clock.tick(FPS)
        timestep += 1
        
        if done:
            running = False
    
    pygame.quit()

if __name__ == "__main__":
    main()
