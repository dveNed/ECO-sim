# ECO-sim v0.5.5
# 
# Changelog (relative to v0.5.4):
# 1) Fixed predator/hunter death counting logic (carried over from v0.5.4).
# 2) Properly separated predator vs herbivore stats in format_stats().
# 3) Implemented 'division' reproduction mechanic: parent's size halved, child inherits the old half.
# 4) Fixed heading in run_away() so fleeing creature faces away from predator.
# 5) Added optional use of eating/fighting countdown timers to prevent "instant repeat" actions.
# 6) Minor code cleanups.

import pygame
import numpy as np
import random
from collections import deque, Counter
from datetime import datetime
import math

# =================== CONFIG ====================
MAX_CREATURES = 1000

INITIAL_HERBIVORES = 180
INITIAL_PREDATORS  = 20

INITIAL_FOOD = 40

AGE_BASE = 700

ENERGY_GAIN_FROM_PREY = 60
ENERGY_GAIN_FROM_FOOD = 15

BASE_ENERGY_LOSS_PER_MOVE = 0.04
SIZE_ENERGY_LOSS_FACTOR   = 0.08

REPRODUCE_ENERGY_COST = 50
ENV_REPRO_CHANCE_BASE = 0.3
ENV_REPRO_FOOD_FACTOR = 0.2
ENV_REPRO_POP_FACTOR  = 0.2

MUTATION_RATE = 0.1
BASE_DIET_FLIP_CHANCE = 0.005

FEAR_MUTATION_CHANCE = 0.2
SIZE_MUTATION_CHANCE = 0.2
LONGEVITY_MUTATION_CHANCE = 0.2

MAX_SPEED = 8.0
MIN_SPEED = 0.5

MIN_SIZE  = 0.5
MAX_SIZE  = 2.0

IDLE_COST = 0.01

EATING_TIME_FRAMES = 15   # 0.5 seconds at 30fps
FIGHTING_TIME_FRAMES = 45 # 1.5 seconds at 30fps

FEAR_COST_BASE = 0.02
FEAR_THRESHOLD_HERB = 0.3

AMBUSH_BONUS = 1.1
PRED_FEAR_THRESHOLD = 0.5
SIZE_DOMINANCE_RATIO = 1.2
HERB_SIZE_DOMINANCE_RATIO = 1.2

SIGHT_RANGE = 200.0
BLOCKING_RADIUS = 5.0

DAY_LENGTH = 600
DAY_COLOR = (60, 140, 180)    # Darker day
NIGHT_COLOR = (0, 10, 35)     # Near-black night

DAYTIME_FOOD_SPAWN_CHANCE = 0.02

LOG_INTERVAL = 30
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"sim_log_{timestamp}.txt"

pygame.init()
screen = pygame.display.set_mode((1200, 800))
font = pygame.font.Font(None, 24)
clock = pygame.time.Clock()

FOOD_COLOR = (242, 243, 244)
GRAPH_COLOR = (200, 100, 150)
PREDATOR_GRAPH_COLOR = (150, 100, 200)

GRID_CELL_SIZE = 25
COLLISION_PASSES = 2

def format_sim_time(frames):
    """Convert frame count to hours:minutes:seconds"""
    total_seconds = frames // 30
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# CHANGE: a revised format_stats with separate herb/pred calculations.
def format_stats(creatures, foods, day_count, sim_frames, death_log, avg_frame_time, mutation_tracker):
    # Population Stats
    live_creatures = [c for c in creatures if c and not c.dead]
    pred_count = sum(1 for c in live_creatures if c.diet == 1)
    herb_count = len(live_creatures) - pred_count
    ratio_str = "∞" if pred_count == 0 else f"{herb_count/pred_count:.1f}"

    # Calculate separate stats for herbs/preds
    if herb_count > 0:
        herb_speed = np.mean([c.speed for c in live_creatures if c.diet == 0])
        herb_size  = np.mean([c.size_factor for c in live_creatures if c.diet == 0])
        herb_fear  = np.mean([c.fear for c in live_creatures if c.diet == 0])
    else:
        herb_speed = herb_size = herb_fear = 0.0

    if pred_count > 0:
        pred_speed = np.mean([c.speed for c in live_creatures if c.diet == 1])
        pred_size  = np.mean([c.size_factor for c in live_creatures if c.diet == 1])
        pred_fear  = np.mean([c.fear for c in live_creatures if c.diet == 1])
    else:
        pred_speed = pred_size = pred_fear = 0.0

    stats = [
        "=== Population ===",
        f"Total: {len(live_creatures)}/{MAX_CREATURES}",
        f"H: {herb_count} | P: {pred_count} (1:{ratio_str})",
        f"Food: {len(foods)} | Day: {day_count}",
        "",
        "=== Herbivores ===",
        f"Speed: {herb_speed:.2f} | Size: {herb_size:.2f} | Fear: {herb_fear:.2f}",
        "",
        "=== Predators ===",
        f"Speed: {pred_speed:.2f} | Size: {pred_size:.2f} | Fear: {pred_fear:.2f}",
        "",
        "=== Time & Deaths ===",
        f"Time: {format_sim_time(sim_frames)}",
        f"S:{death_log['starved']} A:{death_log['aged']} H:{death_log['hunted']}",
        "",
        "[SPACE] Pause/Resume"
    ]
    return stats

def write_log(message):
    if sim_frames % LOG_INTERVAL == 0:
        with open(LOG_FILE, "a") as f:
            f.write(f"[Frame {sim_frames}] {message}\n")

def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def line_distance_point(px, py, ax, ay, bx, by):
    ABx = bx - ax
    ABy = by - ay
    APx = px - ax
    APy = py - ay
    AB_len_sq = ABx*ABx + ABy*ABy
    if AB_len_sq == 0:
        return distance(px, py, ax, ay)
    t = (APx*ABx + APy*ABy) / AB_len_sq
    t = max(0, min(1, t))
    projx = ax + t * ABx
    projy = ay + t * ABy
    return distance(px, py, projx, projy)

def interpolate_color(c1, c2, alpha):
    return (
        int(c1[0] + (c2[0] - c1[0]) * alpha),
        int(c1[1] + (c2[1] - c1[1]) * alpha),
        int(c1[2] + (c2[2] - c1[2]) * alpha),
    )

def calculate_creature_color(energy, speed, fear):
    """Calculate creature color based on attributes (R=energy, G=speed, B=fear)."""
    r = int(np.clip((energy / 100.0) * 255, 50, 255))
    g = int(np.clip((speed / MAX_SPEED) * 255, 50, 255))
    b = int(np.clip(fear * 255, 50, 255))
    return (r, g, b)

# =============== Collisions using Uniform Grid ===============
def collision_broad_phase_grid(creatures):
    grid_map = {}
    for i, c in enumerate(creatures):
        if c is None:
            continue
        cell_x = int(c.x // GRID_CELL_SIZE)
        cell_y = int(c.y // GRID_CELL_SIZE)
        key = (cell_x, cell_y)
        grid_map.setdefault(key, []).append(i)
    return grid_map

def collision_narrow_phase(creatures, indices):
    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            idxA = indices[i]
            idxB = indices[j]
            c1 = creatures[idxA]
            c2 = creatures[idxB]
            if c1 is None or c2 is None:
                continue
            r1 = c1.radius()
            r2 = c2.radius()
            dx = c2.x - c1.x
            dy = c2.y - c1.y
            dist = math.hypot(dx, dy)
            min_dist = r1 + r2
            if dist < min_dist and dist > 0:
                overlap = 0.5 * (min_dist - dist)
                nx = dx / dist
                ny = dy / dist
                c1.x -= overlap * nx
                c1.y -= overlap * ny
                c2.x += overlap * nx
                c2.y += overlap * ny

def resolve_collisions(creatures):
    neighbor_offsets = [
        (0,0),(1,0),(0,1),(1,1),
        (-1,0),(0,-1),(-1,-1),(1,-1),(-1,1)
    ]
    for _ in range(COLLISION_PASSES):
        grid_map = collision_broad_phase_grid(creatures)
        for cell_key, idx_list in grid_map.items():
            collision_narrow_phase(creatures, idx_list)
            for (nx, ny) in neighbor_offsets:
                neigh_key = (cell_key[0]+nx, cell_key[1]+ny)
                if neigh_key in grid_map:
                    combined = idx_list + grid_map[neigh_key]
                    collision_narrow_phase(creatures, combined)

# =====================================================================
class Food:
    def __init__(self):
        self.x = random.randint(50, 1150)
        self.y = random.randint(50, 750)

class Creature:
    """
    v0.5.5 changes:
     - Added optional usage of eating/fighting_countdown
     - 'Division' reproduction (parent/child each half parent's old size)
     - run_away heading fix
     - fix stats to separate herb/pred
    """
    def __init__(self, x, y, father_dna=None):
        self.x = x
        self.y = y
        self.energy = 100
        self.age = 0
        self.heading = random.random() * 2.0 * math.pi
        self.dead = False

        # CHANGE: integrate countdowns (if you want to use them fully)
        self.eating_countdown = 0
        self.fighting_countdown = 0

        if father_dna:
            speed = np.clip(father_dna[0] * random.uniform(0.9, 1.1), MIN_SPEED, MAX_SPEED)
            fear  = father_dna[1]
            if random.random() < FEAR_MUTATION_CHANCE:
                fear += random.uniform(-0.05, 0.05)
                fear = min(max(0.0, fear), 1.0)

            size_factor = father_dna[2]
            if random.random() < SIZE_MUTATION_CHANCE:
                size_factor *= random.uniform(0.9, 1.1)
                size_factor = min(max(MIN_SIZE, size_factor), MAX_SIZE)

            longevity = father_dna[3]
            if random.random() < LONGEVITY_MUTATION_CHANCE:
                longevity *= random.uniform(0.8, 1.2)
                longevity = min(max(0.5, longevity), 2.0)

            diet = father_dna[4]

            self.dna = [speed, fear, size_factor, longevity, diet]

            if random.random() < MUTATION_RATE:
                self.dna[0] *= random.uniform(0.5, 2.0)
                self.dna[0] = np.clip(self.dna[0], MIN_SPEED, MAX_SPEED)
        else:
            self.dna = [
                random.uniform(1.5, 3.0),
                random.uniform(0.0, 0.2),
                random.uniform(MIN_SIZE, 1.2),
                random.uniform(0.8, 1.2),
                0
            ]

        self.speed       = self.dna[0]
        self.fear        = self.dna[1]
        self.size_factor = self.dna[2]
        self.longevity   = self.dna[3]
        self.diet        = int(self.dna[4])

        self.update_color()

    def update_color(self):
        """Update creature color based on current attributes"""
        self.color = calculate_creature_color(self.energy, self.speed, self.fear)

    def _get_mutation_type(self, father_dna):
        if not father_dna:
            return "initial"
        changes = []
        if abs(self.dna[0] - father_dna[0]) > 1.0:
            changes.append("speed_shift")
        if abs(self.dna[1] - father_dna[1]) > 0.05:
            changes.append("fear_shift")
        if abs(self.dna[2] - father_dna[2]) > 0.3:
            changes.append("size_shift")
        if abs(self.dna[3] - father_dna[3]) > 0.3:
            changes.append("long_shift")
        if self.dna[4] != father_dna[4]:
            changes.append("diet_flip")
        if not changes:
            return "minor"
        return "+".join(changes)

    def max_age(self):
        return AGE_BASE * self.longevity

    def strength(self):
        return self.size_factor * (1.0 + 0.5*(self.energy/100.0))

    def max_energy_capacity(self):
        return 100 + (self.size_factor - 1.0)*50

    def radius(self):
        return 6 * self.size_factor

    def update(self, all_creatures, foods):
        """Main logic each frame."""
        if self.dead:
            return ("starved", None)

        self.age += 1
        self.energy -= IDLE_COST

        # CHANGE: if we're 'busy' eating or fighting, skip other actions.
        if self.eating_countdown > 0:
            self.eating_countdown -= 1
            return ("eating", None)
        if self.fighting_countdown > 0:
            self.fighting_countdown -= 1
            return ("fighting", None)

        # Fear cost
        if self.diet == 0 and self.fear > FEAR_THRESHOLD_HERB:
            self.energy -= FEAR_COST_BASE * self.fear * self.size_factor
        elif self.diet == 1 and self.fear > PRED_FEAR_THRESHOLD:
            self.energy -= FEAR_COST_BASE * self.fear * (self.size_factor * 0.5)

        # Movement cost
        self.energy -= BASE_ENERGY_LOSS_PER_MOVE
        self.energy -= (self.size_factor * SIZE_ENERGY_LOSS_FACTOR)
        self.energy = max(self.energy, 0)

        self.update_color()

        if self.diet == 0:
            # Herbivore logic
            if self.fear > FEAR_THRESHOLD_HERB:
                predator, distp = self.find_nearest(all_creatures, lambda c: c is not self and c.diet == 1 and not c.dead)
                if predator and distp < 120:
                    self.run_away(predator.x, predator.y)
                    self.wrap_screen()
                    return self.check_survival()

            food_t, distf = self.find_nearest(foods)
            if food_t and distf < SIGHT_RANGE:
                if not self.is_line_blocked(food_t.x, food_t.y, all_creatures):
                    self.move_toward(food_t.x, food_t.y)
                    if distance(self.x, self.y, food_t.x, food_t.y) < (6 + self.size_factor*4):
                        self.energy = min(self.max_energy_capacity(), self.energy + ENERGY_GAIN_FROM_FOOD)
                        foods.remove(food_t)
                        # CHANGE: set countdown if we want a short 'eating' pause
                        self.eating_countdown = EATING_TIME_FRAMES
                else:
                    self.random_walk()
            else:
                self.random_walk()

        else:
            # Predator logic
            ctarget, distc = self.find_nearest(all_creatures, lambda c: c is not self and not c.dead)
            if ctarget and distc < SIGHT_RANGE:
                if not self.is_line_blocked(ctarget.x, ctarget.y, all_creatures):
                    if ctarget.diet == 1:  # Another predator
                        if self.fear > PRED_FEAR_THRESHOLD and (ctarget.size_factor > self.size_factor*SIZE_DOMINANCE_RATIO):
                            self.run_away(ctarget.x, ctarget.y)
                        else:
                            self.move_toward(ctarget.x, ctarget.y)
                            if distance(self.x, self.y, ctarget.x, ctarget.y) < (10 + self.size_factor*4 + ctarget.size_factor*4):
                                return self.predator_fight(ctarget)
                    else:  # ctarget.diet == 0
                        if (ctarget.size_factor > self.size_factor*HERB_SIZE_DOMINANCE_RATIO) and (self.fear > PRED_FEAR_THRESHOLD):
                            self.run_away(ctarget.x, ctarget.y)
                        else:
                            self.move_toward(ctarget.x, ctarget.y)
                            if distance(self.x, self.y, ctarget.x, ctarget.y) < (10 + self.size_factor*4 + ctarget.size_factor*4):
                                gain = ENERGY_GAIN_FROM_PREY * ctarget.size_factor
                                self.energy = min(self.max_energy_capacity(), self.energy + gain)
                                ctarget.dead = True
                                # CHANGE: short fighting pause
                                self.fighting_countdown = FIGHTING_TIME_FRAMES
                                return ("ate_creature", ctarget)
                else:
                    self.random_walk()
            else:
                self.random_walk()

        self.wrap_screen()
        return self.check_survival()

    def predator_fight(self, other):
        """Predator vs predator fight logic."""
        if other.dead:
            return ("alive", None)

        # Possibly give attacking advantage ~ 30% of the time
        if random.random() < 0.3:
            my_str = self.strength() * AMBUSH_BONUS
            their_str = other.strength()
        else:
            my_str = self.strength()
            their_str = other.strength()

        if my_str > their_str:
            gain = ENERGY_GAIN_FROM_PREY * other.size_factor
            self.energy = min(self.max_energy_capacity(), self.energy + gain)
            other.dead = True
            self.fighting_countdown = FIGHTING_TIME_FRAMES
            if other.diet == 0:
                return ("ate_creature", other)
            return ("pred_fight", other)
        elif their_str > my_str:
            self.dead = True
            if self.diet == 0:
                return ("ate_creature", self)
            return ("pred_fight", self)
        else:
            if random.random() < 0.5:
                other.dead = True
                self.fighting_countdown = FIGHTING_TIME_FRAMES
                if other.diet == 0:
                    return ("ate_creature", other)
                return ("pred_fight", other)
            else:
                self.dead = True
                if self.diet == 0:
                    return ("ate_creature", self)
                return ("pred_fight", self)

    def check_survival(self):
        """Check if we died from zero energy or old age."""
        if self.dead:
            return ("starved", None)
        if self.energy <= 0:
            self.dead = True
            return ("starved", None)
        if self.age > self.max_age():
            self.dead = True
            return ("aged", None)
        return ("alive", None)

    # CHANGE: halved size in reproduction => 'division' style
    def reproduce(self, creatures, foods):
        if self.dead or self.energy < 80 or self.size_factor < 0.7:
            return None

        # Save old size to pass to child
        old_size = self.size_factor
        # Halve the parent's size
        self.size_factor *= 0.5

        pop_size = len(creatures)
        food_count = len(foods)
        ratio_food_pop = food_count / pop_size if pop_size > 0 else 1
        ratio_pop_cap = pop_size / MAX_CREATURES

        chance = ENV_REPRO_CHANCE_BASE
        chance += ENV_REPRO_FOOD_FACTOR * ratio_food_pop
        chance -= ENV_REPRO_POP_FACTOR * ratio_pop_cap
        chance = min(max(chance, 0.0), 1.0)

        if random.random() < chance:
            self.energy -= REPRODUCE_ENERGY_COST
            child_dna = list(self.dna)
            
            # Possibly flip diet if environment is tough
            herb_count = sum(1 for c in creatures if c and c.diet == 0 and not c.dead)
            pred_count = pop_size - herb_count
            flip_chance = BASE_DIET_FLIP_CHANCE

            if self.diet == 0:
                if food_count < (herb_count * 0.7):
                    flip_chance += 0.05
                if (self.size_factor > 1.5 and ratio_food_pop < 0.5):
                    flip_chance += 0.05
                if random.random() < flip_chance:
                    child_dna[4] = 1
            else:
                if pred_count > (herb_count * 0.75):
                    flip_chance += 0.05
                if random.random() < flip_chance:
                    child_dna[4] = 0

            # Give child the 'old size' (the parent's size pre-halving)
            child_dna[2] = old_size * 0.5

            offx = random.gauss(0, 10)
            offy = random.gauss(0, 10)
            baby = Creature(self.x + offx, self.y + offy, father_dna=child_dna)
            return baby
        return None

    def random_walk(self):
        self.heading += random.uniform(-0.2, 0.2)
        dx = math.cos(self.heading) * self.speed
        dy = math.sin(self.heading) * self.speed
        self.x += dx
        self.y += dy

    # CHANGE: fix heading so that it *faces away* from the threat
    def run_away(self, ox, oy):
        dx = self.x - ox
        dy = self.y - oy
        distv = distance(self.x, self.y, ox, oy)
        if distv > 1:
            self.x += (dx/distv) * self.speed
            self.y += (dy/distv) * self.speed
            # Face away from predator
            self.heading = math.atan2(- (oy - self.y), - (ox - self.x))
        else:
            self.random_walk()

    def move_toward(self, tx, ty):
        dx = tx - self.x
        dy = ty - self.y
        distv = distance(self.x, self.y, tx, ty)
        if distv > 1:
            self.x += (dx/distv) * self.speed
            self.y += (dy/distv) * self.speed
            self.heading = math.atan2(dy, dx)
        else:
            self.random_walk()

    def wrap_screen(self):
        self.x %= 1200
        self.y %= 800

    def find_nearest(self, target_list, condition=None):
        valids = [t for t in target_list if (t and (t is not self))]
        if condition:
            valids = [v for v in valids if condition(v)]
        if not valids:
            return (None, float('inf'))
        best_obj, best_dist = None, float('inf')
        for obj in valids:
            d = distance(self.x, self.y, obj.x, obj.y)
            if d < best_dist:
                best_dist = d
                best_obj = obj
        return (best_obj, best_dist)

    def is_line_blocked(self, tx, ty, creatures):
        targ_dist = distance(self.x, self.y, tx, ty)
        for c in creatures:
            if not c or c is self or c.dead:
                continue
            dist_c = distance(self.x, self.y, c.x, c.y)
            if dist_c >= targ_dist:
                continue
            d_line = line_distance_point(c.x, c.y, self.x, self.y, tx, ty)
            if d_line < BLOCKING_RADIUS:
                return True
        return False

    def draw(self, surface):
        if self.dead:
            return
        sz = int(8 * self.size_factor)
        if self.diet == 0:
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), sz)
        else:
            # Predator "cursor" shape
            tip_x = self.x + math.cos(self.heading) * sz * 2.5
            tip_y = self.y + math.sin(self.heading) * sz * 2.5
            base_x = self.x - math.cos(self.heading) * sz * 0.5
            base_y = self.y - math.sin(self.heading) * sz * 0.5
            perp_x = math.cos(self.heading + math.pi/2) * sz * 0.4
            perp_y = math.sin(self.heading + math.pi/2) * sz * 0.4

            points = [
                (tip_x,                tip_y),
                (base_x + perp_x,      base_y + perp_y),
                (base_x,               base_y),
                (base_x - perp_x,      base_y - perp_y)
            ]
            pygame.draw.polygon(surface, self.color, points)

# ==================== SETUP =========================
open(LOG_FILE, "w").close()

creatures = []
for _ in range(INITIAL_HERBIVORES):
    c = Creature(random.randint(100, 1100), random.randint(100, 700))
    c.dna[4] = 0
    c.diet = 0
    creatures.append(c)

for _ in range(INITIAL_PREDATORS):
    c = Creature(random.randint(100, 1100), random.randint(100, 700))
    c.dna[4] = 1
    c.diet = 1
    creatures.append(c)

foods = [Food() for _ in range(INITIAL_FOOD)]

paused = False
frames = 0
sim_frames = 0
population_history = deque(maxlen=300)
pred_ratio_history = deque(maxlen=300)

death_log = {'starved': 0, 'aged': 0, 'hunted': 0}
mutation_tracker = Counter()

day_cycle_frame = 0
day_count = 0
fps_sample_timer = 0.0
avg_frame_time = 0.0

running = True

while running:
    dt = clock.tick(30)
    frames += 1

    fps_sample_timer += dt
    avg_frame_time = 0.9*avg_frame_time + 0.1*dt

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused

    if not paused:
        sim_frames += 1
        day_cycle_frame = (day_cycle_frame + 1) % DAY_LENGTH
        if day_cycle_frame == 0:
            day_count += 1

    alpha = day_cycle_frame/(DAY_LENGTH-1) if (DAY_LENGTH > 1) else 1.0
    if alpha < 0.5:
        sub_a = alpha/0.5
        bg_color = interpolate_color(NIGHT_COLOR, DAY_COLOR, sub_a)
        is_day = True
    else:
        sub_a = (alpha-0.5)/0.5
        bg_color = interpolate_color(DAY_COLOR, NIGHT_COLOR, sub_a)
        is_day = False

    screen.fill(bg_color)

    if not paused:
        new_creatures = []
        dead = []

        for c in creatures:
            if c and not c.dead:
                status, other = c.update(creatures, foods)

                if status == "starved":
                    death_log["starved"] += 1
                    write_log(f"Death: starved (speed={c.speed:.2f})")
                    dead.append(c)
                    food_chunks = 1 + int(c.size_factor*2)
                    for _ in range(food_chunks):
                        if random.random() < 0.3:
                            foods.append(Food())

                elif status == "aged":
                    death_log["aged"] += 1
                    write_log(f"Death: aged (speed={c.speed:.2f})")
                    dead.append(c)
                    food_chunks = 1 + int(c.size_factor*2)
                    for _ in range(food_chunks):
                        if random.random() < 0.3:
                            foods.append(Food())

                elif status == "ate_creature":
                    if other and other.diet == 0:  # Only count herbivore as 'hunted'
                        death_log["hunted"] += 1
                    reason = "hunted" if (other and other.diet == 0) else "pred_fight"
                    if other:
                        write_log(f"Death: {reason} (speed={other.speed:.2f})")
                        dead.append(other)
                        food_chunks = 1 + int(other.size_factor*2)
                        for _ in range(food_chunks):
                            if random.random() < 0.3:
                                foods.append(Food())

                elif status == "pred_fight":
                    # Predator killed by predator
                    if other:
                        write_log(f"Death: pred_fight (speed={other.speed:.2f})")
                        dead.append(other)
                        food_chunks = 1 + int(other.size_factor*2)
                        for _ in range(food_chunks):
                            if random.random() < 0.3:
                                foods.append(Food())

                # "alive", "eating", "fighting" just continue

                # Attempt reproduction
                child = c.reproduce(creatures, foods)
                if child and len(creatures) < MAX_CREATURES:
                    mut_type = child._get_mutation_type(c.dna)
                    if mut_type not in ("initial", "minor"):
                        mutation_tracker[mut_type] += 1

                    new_creatures.append(child)
                    write_log(
                        f"Birth: speed={child.speed:.2f}, diet={child.diet}, "
                        f"size={child.size_factor:.2f}, fear={child.fear:.2f}, "
                        f"longevity={child.longevity:.2f}, mutation={mut_type}"
                    )

        for dcreat in dead:
            if dcreat in creatures:
                idx = creatures.index(dcreat)
                creatures[idx] = None

        creatures = [cr for cr in creatures if cr is not None and not cr.dead]
        space_left = max(0, MAX_CREATURES - len(creatures))
        creatures += new_creatures[:space_left]

        resolve_collisions(creatures)

        pop_size = len(creatures)
        pred_count = sum(1 for cr in creatures if cr and cr.diet == 1 and not cr.dead)
        population_history.append(pop_size)
        ratio = pred_count/pop_size if pop_size > 0 else 0
        pred_ratio_history.append(ratio)

        if is_day:
            if random.random() < DAYTIME_FOOD_SPAWN_CHANCE:
                foods.append(Food())

    # Draw
    for f in foods:
        pygame.draw.circle(screen, FOOD_COLOR, (int(f.x), int(f.y)), 4)

    for cr in creatures:
        if cr and not cr.dead:
            cr.draw(screen)

    stats = format_stats(creatures, foods, day_count, sim_frames, death_log, avg_frame_time, mutation_tracker)
    for i, text in enumerate(stats):
        txt = font.render(text, True, (255, 255, 255))
        screen.blit(txt, (10, 10 + i*25))

    # Show top mutations
    mut_top3 = mutation_tracker.most_common(3)
    if mut_top3:
        mut_strs = [f"{mt}:{cnt}" for (mt, cnt) in mut_top3]
        mut_line = "=== Mutations === " + ", ".join(mut_strs)
        txtm = font.render(mut_line, True, (255, 255, 255))
        screen.blit(txtm, (10, 10 + len(stats)*25))

    # Population Graph
    graph_x, graph_y = 800, 70
    graph_w, graph_h = 380, 90
    pygame.draw.rect(screen, (50, 50, 50), (graph_x, graph_y, graph_w, graph_h))

    if len(population_history) > 1:
        points_pop = []
        for i, val in enumerate(population_history):
            xx = graph_x + graph_w - (len(population_history) - i)*2
            yy = graph_y + graph_h - ((val/MAX_CREATURES)*graph_h)
            points_pop.append((xx, yy))
        pygame.draw.lines(screen, GRAPH_COLOR, False, points_pop, 2)

    # Predator ratio graph
    if len(pred_ratio_history) > 1:
        points_pred = []
        for i, ratio_val in enumerate(pred_ratio_history):
            xx = graph_x + graph_w - (len(pred_ratio_history) - i)*2
            yy = graph_y + graph_h - (ratio_val*graph_h)
            points_pred.append((xx, yy))
        pygame.draw.lines(screen, PREDATOR_GRAPH_COLOR, False, points_pred, 2)

    pygame.display.flip()

pygame.quit()
