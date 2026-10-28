import math
import sys
import numpy as np
import pygame

# ============================================================
# JOGO DE PLATAFORMA 2D
# Álgebra Linear aplicada a Jogos
#
# Dependências:
#   pip install pygame numpy
#
# Controles:
#   A / D ou setas: mover
#   W / Espaço: pular
#   R: reiniciar
#   ESC: sair
#
# Conceitos de Álgebra Linear presentes:
#   - vetores em R^2
#   - base canônica
#   - soma de vetores
#   - multiplicação por escalar
#   - norma e distância
#   - normalização
#   - produto interno
#   - matriz de rotação
#   - transformação linear
#
# Acrescentados nas melhorias:
#   - coordenadas homogêneas (translação como matriz 3x3)
#   - composição de transformações (C . T . R . S)
#   - matriz de escala não uniforme com det = 1
#   - projeção ortogonal sobre E1
#   - matriz de reflexão de Householder, H = I - 2 n n^T
# ============================================================

pygame.init()

WIDTH, HEIGHT = 1000, 600
FPS = 60

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Álgebra Linear - Platform Hunter")
clock = pygame.time.Clock()

font = pygame.font.Font(None, 28)
small_font = pygame.font.Font(None, 22)
title_font = pygame.font.Font(None, 36)

# ------------------------------------------------------------
# Mundo
# ------------------------------------------------------------

WORLD_WIDTH = 2400
WORLD_HEIGHT = 900
GRAVITY = 1200.0

# Base canônica de R^2:
E1 = np.array([1.0, 0.0])
E2 = np.array([0.0, 1.0])

# ------------------------------------------------------------
# Plataformas
# Cada plataforma: x, y, largura, altura
# ------------------------------------------------------------

platforms = [
    pygame.Rect(0, 540, 700, 60),
    pygame.Rect(780, 470, 260, 40),
    pygame.Rect(1120, 390, 260, 40),
    pygame.Rect(1450, 500, 300, 40),
    pygame.Rect(1780, 410, 280, 40),
    pygame.Rect(2120, 520, 280, 60),
    pygame.Rect(900, 650, 300, 35),
    pygame.Rect(1320, 700, 280, 35),
]

goal_rect = pygame.Rect(2320, 440, 30, 80)

# Cristais coletáveis: o centro de cada um é um vetor de R^2.
crystals = [
    np.array([420.0, 500.0]),
    np.array([910.0, 430.0]),
    np.array([1250.0, 350.0]),
    np.array([1600.0, 460.0]),
    np.array([1920.0, 370.0]),
    np.array([2200.0, 480.0]),
]

# O jogador pega o cristal quando ||centro - cristal|| <= raio.
CRYSTAL_RADIUS = 30.0

# ------------------------------------------------------------
# Cores
# ------------------------------------------------------------

BG = (25, 30, 42)
PLATFORM_COLOR = (85, 105, 130)
PLAYER_COLOR = (70, 180, 255)
ENEMY_COLOR = (235, 80, 90)
TEXT_COLOR = (235, 240, 245)
VISION_COLOR = (240, 190, 70)
GOAL_COLOR = (90, 220, 120)
PROJECTILE_COLOR = (255, 150, 90)
CRYSTAL_COLOR = (170, 120, 255)

# ------------------------------------------------------------
# Funções de Álgebra Linear
# ------------------------------------------------------------

def normalize(v):
    """Retorna v / ||v||. Para vetor nulo, retorna o próprio vetor."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v.copy()
    return v / norm


def rotation_matrix(theta):
    """Matriz de rotação 2D."""
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([
        [c, -s],
        [s,  c]
    ], dtype=float)


def reflection_matrix(n):
    """
    Matriz de reflexão de Householder: H = I - 2 n n^T, com n unitário.
    H v tira de v duas vezes a sua projeção sobre n, o que espelha v
    em relação à superfície de normal n.
    """
    n = normalize(n)
    return np.eye(2) - 2.0 * np.outer(n, n)


def scale_matrix(sx, sy):
    """Matriz de escala 2D (diagonal). det = sx * sy."""
    return np.array([
        [sx, 0.0],
        [0.0, sy]
    ], dtype=float)


def translation_matrix(tx, ty):
    """
    Translação em coordenadas homogêneas.
    Em R^2 a translação não é linear (leva a origem para (tx, ty)),
    mas em R^3, com o ponto escrito como (x, y, 1), ela vira uma
    multiplicação de matriz.
    """
    return np.array([
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
        [0.0, 0.0, 1.0]
    ], dtype=float)


def to_homogeneous(A):
    """Coloca uma matriz 2x2 no canto de uma matriz 3x3."""
    H = np.eye(3)
    H[:2, :2] = A
    return H


def model_matrix(position, angle=0.0, sx=1.0, sy=1.0):
    """
    M = T(position) . R(angle) . S(sx, sy)
    Lida da direita para a esquerda: primeiro escala, depois
    rotaciona e por último translada.
    """
    A = rotation_matrix(angle) @ scale_matrix(sx, sy)
    return translation_matrix(position[0], position[1]) @ to_homogeneous(A)


def apply_matrix(M, points):
    """
    Aplica a matriz 3x3 M a uma lista de pontos 2D de uma vez:
    os pontos viram as colunas de uma matriz 3 x n.
    """
    P = np.vstack([np.array(points, dtype=float).T, np.ones(len(points))])
    Q = M @ P
    return [tuple(q) for q in Q[:2].T]


# ------------------------------------------------------------
# Classe Jogador
# ------------------------------------------------------------

class Player:
    def __init__(self):
        self.reset()

    def reset(self):
        self.pos = np.array([120.0, 450.0], dtype=float)
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.width = 36
        self.height = 50
        self.speed = 260.0
        self.jump_speed = -600.0
        self.on_ground = False
        self.collected = set()
        self.squash = 0.0

    @property
    def rect(self):
        return pygame.Rect(
            int(self.pos[0]),
            int(self.pos[1]),
            self.width,
            self.height
        )

    @property
    def center(self):
        return self.pos + np.array([
            self.width / 2,
            self.height / 2
        ])

    def update(self, dt, keys):
        # Vetor de movimento horizontal.
        move = 0.0

        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            move -= 1.0
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            move += 1.0

        self.velocity[0] = move * self.speed

        if (keys[pygame.K_w] or keys[pygame.K_SPACE]
                or keys[pygame.K_UP]) and self.on_ground:
            self.velocity[1] = self.jump_speed
            self.on_ground = False

        # Gravidade como vetor.
        gravity = np.array([0.0, GRAVITY])
        self.velocity += gravity * dt

        # Atualização de posição:
        # p(t + dt) = p(t) + v * dt
        self.pos += self.velocity * dt

        fall_speed = self.velocity[1]
        self._resolve_collisions(dt)

        # Pouso forte achata o desenho, que volta ao normal aos poucos.
        if self.on_ground and fall_speed > 300:
            self.squash = min(0.3, fall_speed / 3000.0)
        self.squash = max(0.0, self.squash - 1.5 * dt)

        # Limites horizontais do mundo.
        self.pos[0] = max(0, min(self.pos[0], WORLD_WIDTH - self.width))

        # Queda fora do mundo.
        if self.pos[1] > WORLD_HEIGHT:
            self.reset()

        # Coleta por distância: norma do vetor centro -> cristal.
        for i, crystal in enumerate(crystals):
            if (i not in self.collected
                    and np.linalg.norm(self.center - crystal) <= CRYSTAL_RADIUS):
                self.collected.add(i)

    def scale_factors(self):
        """
        Escala não uniforme S = diag(sx, sy) com sx = 1 / sy, então
        det(S) = sx * sy = 1: o desenho muda de forma mas mantém a área.
        No ar estica na vertical; ao pousar achata.
        """
        sy = 1.0 + min(abs(self.velocity[1]) / 2400.0, 0.2) - self.squash
        return 1.0 / sy, sy

    def _resolve_collisions(self, dt):
        rect = self.rect
        self.on_ground = False

        # Colisão vertical simplificada.
        for platform in platforms:
            if rect.colliderect(platform):
                # Posição anterior = posição atual - v * dt. O original
                # usava 1 / FPS, que só vale se o jogo rodar a 60 FPS.
                previous_bottom = self.pos[1] + self.height - self.velocity[1] * dt

                # Pousando sobre a plataforma.
                if self.velocity[1] >= 0 and previous_bottom <= platform.top + 8:
                    self.pos[1] = platform.top - self.height
                    self.velocity[1] = 0
                    self.on_ground = True
                    rect = self.rect

                # Batendo na parte inferior.
                elif self.velocity[1] < 0 and self.pos[1] >= platform.bottom - 5:
                    self.pos[1] = platform.bottom
                    self.velocity[1] = 0
                    rect = self.rect


# ------------------------------------------------------------
# Projétil do inimigo
# ------------------------------------------------------------

def collision_normal(pos, radius, rect):
    """
    Normal unitária da face do retângulo em que o círculo entrou
    menos. As normais são vetores da base canônica: -E1, E1, -E2, E2.
    Retorna a profundidade da penetração e a normal.
    """
    penetrations = [
        (pos[0] + radius - rect.left, -E1),
        (rect.right - (pos[0] - radius), E1),
        (pos[1] + radius - rect.top, -E2),
        (rect.bottom - (pos[1] - radius), E2),
    ]
    return min(penetrations, key=lambda item: item[0])


class Projectile:
    RADIUS = 6
    SPEED = 280.0
    MAX_BOUNCES = 2
    LIFETIME = 3.0

    def __init__(self, pos, direction_unit):
        self.pos = pos.copy()
        # Velocidade = escalar * vetor unitário.
        self.velocity = self.SPEED * direction_unit
        self.bounces = 0
        self.age = 0.0

    @property
    def rect(self):
        r = self.RADIUS
        return pygame.Rect(int(self.pos[0] - r), int(self.pos[1] - r), 2 * r, 2 * r)

    def update(self, dt):
        """Move o projétil e retorna False quando ele deve sumir."""
        self.age += dt
        self.pos += self.velocity * dt

        for platform in platforms:
            if self.rect.colliderect(platform):
                depth, n = collision_normal(self.pos, self.RADIUS, platform)

                # Empurra o projétil para fora ao longo da normal.
                self.pos += depth * n

                # Só reflete se estiver entrando na superfície (v . n < 0).
                # |H v| = |v|: a reflexão muda a direção, não a rapidez.
                if np.dot(self.velocity, n) < 0:
                    self.velocity = reflection_matrix(n) @ self.velocity
                    self.bounces += 1

        return (
            self.bounces <= self.MAX_BOUNCES
            and self.age < self.LIFETIME
            and 0 <= self.pos[0] <= WORLD_WIDTH
            and self.pos[1] <= WORLD_HEIGHT
        )


# ------------------------------------------------------------
# Classe Inimigo
# ------------------------------------------------------------

class Enemy:
    def __init__(self):
        self.reset()

    def reset(self):
        self.pos = np.array([560.0, 455.0], dtype=float)
        self.velocity = np.array([0.0, 0.0], dtype=float)
        self.max_speed = 115.0
        self.patrol_speed = 50.0
        self.patrol_min = 120.0
        self.patrol_max = 900.0
        self.vision_distance = 420.0
        self.projectiles = []
        self.shot_cooldown = 1.5

        # Campo de visão de 80 graus.
        self.fov_half_angle = math.radians(40.0)
        self.fov_threshold = math.cos(self.fov_half_angle)

        # Direção inicial: o jogador começa à esquerda do inimigo,
        # então o inimigo começa olhando para a esquerda.
        self.forward = np.array([-1.0, 0.0], dtype=float)

        # Ângulo usado na transformação matricial.
        self.angle = math.pi

    @property
    def rect(self):
        return pygame.Rect(
            int(self.pos[0] - 20),
            int(self.pos[1] - 20),
            40,
            40
        )

    def update(self, player, dt):
        # Vetor do inimigo até o jogador.
        direction = player.center - self.pos

        distance = np.linalg.norm(direction)

        if distance > 0:
            direction_unit = direction / distance
        else:
            direction_unit = np.zeros(2)

        # Produto interno entre direção para frente
        # e direção do alvo.
        dot = float(np.dot(self.forward, direction_unit))

        # Decide se o jogador está dentro do campo de visão.
        can_see_player = (
            distance <= self.vision_distance
            and dot >= self.fov_threshold
        )

        if can_see_player:
            # O inimigo persegue o jogador.
            self.velocity = self.max_speed * direction_unit

            # O ângulo acompanha a direção.
            self.angle = math.atan2(
                direction_unit[1],
                direction_unit[0]
            )

            # Atualiza o vetor forward usando rotação.
            self.forward = rotation_matrix(self.angle) @ E1
        else:
            # Patrulha de ida e volta na horizontal.
            # Ao perder o jogador, forward é projetado sobre o eixo E1,
            # proj = (forward . E1) E1, e normalizado: o inimigo segue
            # para o lado em que estava olhando.
            horizontal = np.dot(self.forward, E1) * E1
            if horizontal[0] == 0:
                horizontal = -E1
            self.forward = normalize(horizontal)

            # No limite da patrulha o sentido inverte: forward vira
            # (-1) * forward, uma rotação de 180 graus.
            if ((self.pos[0] <= self.patrol_min and self.forward[0] < 0)
                    or (self.pos[0] >= self.patrol_max and self.forward[0] > 0)):
                self.forward = -self.forward

            # O vetor forward precisa corresponder ao sentido da patrulha.
            self.velocity = self.patrol_speed * self.forward
            self.angle = math.atan2(self.forward[1], self.forward[0])

        self.pos += self.velocity * dt

        # Mantém o inimigo em uma região jogável.
        self.pos[0] = max(self.patrol_min, min(self.pos[0], self.patrol_max))

        # Atira na direção do jogador depois de mirar por um tempo.
        if can_see_player:
            self.shot_cooldown -= dt
        if can_see_player and self.shot_cooldown <= 0:
            self.projectiles.append(
                Projectile(self.pos + 24 * direction_unit, direction_unit)
            )
            self.shot_cooldown = 2.0

        self.projectiles = [p for p in self.projectiles if p.update(dt)]
        hit = any(p.rect.colliderect(player.rect) for p in self.projectiles)

        # Se encostar no jogador ou acertar um tiro, o jogo reinicia.
        if self.rect.colliderect(player.rect) or hit:
            player.reset()
            self.reset()

        return distance, dot, can_see_player


# ------------------------------------------------------------
# Desenho do inimigo usando transformação matricial
# ------------------------------------------------------------

def draw_enemy(surface, enemy, camera):
    # Triângulo no sistema de coordenadas local.
    local_points = [
        (28, 0),
        (-18, -16),
        (-18, 16)
    ]

    # Transformação em coordenadas homogêneas:
    # p_tela = C . T(position) . R(theta) . S . p_local
    # C é a translação da câmera, -camera.
    C = translation_matrix(-camera[0], -camera[1])
    M = C @ model_matrix(enemy.pos, enemy.angle)

    screen_points = apply_matrix(M, local_points)

    pygame.draw.polygon(surface, ENEMY_COLOR, screen_points)


def draw_crystals(surface, player, camera):
    # Losango no sistema local, girando e pulsando com o tempo:
    # p_tela = C . T(centro) . R(t) . S(s, s) . p_local
    local_points = [(0, -12), (9, 0), (0, 12), (-9, 0)]
    t = pygame.time.get_ticks() / 1000.0
    angle = 2.0 * t
    s = 1.0 + 0.15 * math.sin(3.0 * t)

    C = translation_matrix(-camera[0], -camera[1])
    for i, crystal in enumerate(crystals):
        if i in player.collected:
            continue
        M = C @ model_matrix(crystal, angle, s, s)
        pygame.draw.polygon(surface, CRYSTAL_COLOR, apply_matrix(M, local_points))


def draw_projectiles(surface, enemy, camera):
    for p in enemy.projectiles:
        center = p.pos - camera
        # Rastro no sentido oposto ao da velocidade.
        tail = center - 0.06 * p.velocity
        pygame.draw.line(surface, PROJECTILE_COLOR, tuple(tail), tuple(center), 3)
        pygame.draw.circle(surface, PROJECTILE_COLOR, tuple(center), p.RADIUS)


def draw_player(surface, player, camera):
    # A origem local fica nos pés: a escala estica o retângulo para
    # cima e os pés continuam no chão. A colisão usa o retângulo
    # original; a escala é só visual.
    w, h = player.width, player.height
    local_points = [(-w / 2, -h), (w / 2, -h), (w / 2, 0), (-w / 2, 0)]
    feet = player.pos + np.array([w / 2, h])
    sx, sy = player.scale_factors()

    C = translation_matrix(-camera[0], -camera[1])
    M = C @ model_matrix(feet, 0.0, sx, sy)
    pygame.draw.polygon(surface, PLAYER_COLOR, apply_matrix(M, local_points))


def draw_platforms(surface, camera):
    for platform in platforms:
        rect = platform.copy()
        rect.x -= int(camera[0])
        rect.y -= int(camera[1])
        pygame.draw.rect(surface, PLATFORM_COLOR, rect, border_radius=4)


def draw_goal(surface, camera):
    rect = goal_rect.copy()
    rect.x -= int(camera[0])
    rect.y -= int(camera[1])

    pygame.draw.rect(surface, GOAL_COLOR, rect)

    # Bandeira simples.
    pygame.draw.line(
        surface,
        GOAL_COLOR,
        (rect.left, rect.top),
        (rect.left, rect.top - 60),
        4
    )

    pygame.draw.polygon(
        surface,
        GOAL_COLOR,
        [
            (rect.left + 2, rect.top - 58),
            (rect.left + 35, rect.top - 48),
            (rect.left + 2, rect.top - 38),
        ]
    )


# Camada transparente reaproveitada a cada quadro.
vision_layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)


def draw_vision(surface, enemy, camera, distance, dot, can_see):
    # O cone desenhado é a mesma região testada no update():
    #   ||d|| <= vision_distance  e  forward . u >= cos(40°).
    # Cada ponto do arco é o vetor forward girado de um ângulo t
    # entre -40° e +40° e multiplicado pelo raio.
    center = enemy.pos - camera
    radius = enemy.vision_distance
    half = enemy.fov_half_angle

    arc = [
        center + radius * (rotation_matrix(t) @ enemy.forward)
        for t in np.linspace(-half, half, 17)
    ]
    cone = [tuple(center)] + [tuple(p) for p in arc]

    color = ENEMY_COLOR if can_see else VISION_COLOR

    vision_layer.fill((0, 0, 0, 0))
    pygame.draw.polygon(vision_layer, (*color, 45), cone)

    # Circunferência ||d|| = vision_distance (condição da norma).
    pygame.draw.circle(vision_layer, (*VISION_COLOR, 60), tuple(center), radius, 1)
    surface.blit(vision_layer, (0, 0))

    pygame.draw.polygon(surface, color, cone, 1)

    # Direção do inimigo.
    end = center + enemy.forward * radius
    pygame.draw.line(surface, color, tuple(center), tuple(end), 2)


def draw_hud(surface, player, enemy, distance, dot, can_see):
    title = title_font.render(
        "Platform Hunter - Álgebra Linear",
        True,
        TEXT_COLOR
    )
    surface.blit(title, (18, 16))

    sx, sy = player.scale_factors()

    lines = [
        f"Player position = ({player.pos[0]:.1f}, {player.pos[1]:.1f})",
        f"Player velocity = ({player.velocity[0]:.1f}, {player.velocity[1]:.1f})",
        f"Enemy position  = ({enemy.pos[0]:.1f}, {enemy.pos[1]:.1f})",
        f"distance = {distance:.2f}",
        f"forward . direction_unit = {dot:.3f}",
        f"angulo = arccos(dot) = {math.degrees(math.acos(max(-1.0, min(1.0, dot)))):.1f} graus (limite 40)",
        f"field of view = {'SIM' if can_see else 'NAO'}",
        f"projeteis = {len(enemy.projectiles)}   (reflexao: v' = (I - 2nn^T) v)",
        f"cristais = {len(player.collected)}/{len(crystals)}   (coleta: ||p - c|| <= {CRYSTAL_RADIUS:.0f})",
        f"escala do jogador: sx = {sx:.2f}, sy = {sy:.2f}, det = {sx * sy:.2f}",
        "A/D ou setas: mover   |   W/Espaco: pular   |   R: reiniciar",
    ]

    y = 58
    for line in lines:
        text = small_font.render(line, True, TEXT_COLOR)
        surface.blit(text, (18, y))
        y += 23


def draw_background(surface, camera):
    surface.fill(BG)

    # Grade para reforçar a interpretação das posições como vetores.
    grid_size = 50
    start_x = -(int(camera[0]) % grid_size)
    start_y = -(int(camera[1]) % grid_size)

    for x in range(start_x, WIDTH, grid_size):
        pygame.draw.line(surface, (38, 44, 58), (x, 0), (x, HEIGHT), 1)

    for y in range(start_y, HEIGHT, grid_size):
        pygame.draw.line(surface, (38, 44, 58), (0, y), (WIDTH, y), 1)


# ------------------------------------------------------------
# Loop principal
# ------------------------------------------------------------

def main():
    player = Player()
    enemy = Enemy()

    win_text = ""
    win_timer = 0.0

    running = True

    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    player.reset()
                    enemy.reset()

        keys = pygame.key.get_pressed()

        player.update(dt, keys)
        distance, dot, can_see = enemy.update(player, dt)

        # Vitória.
        if player.rect.colliderect(goal_rect):
            win_text = f"Vitória! Cristais:{len(player.collected)}/{len(crystals)}"
            win_timer = 2.5
            player.reset()
            enemy.reset()

        # Câmera acompanha o jogador.
        camera_x = player.pos[0] - WIDTH * 0.35
        camera_x = max(0, min(camera_x, WORLD_WIDTH - WIDTH))

        # Na vertical a câmera só desce quando os pés do jogador passam
        # de HEIGHT - 50. Sem isso, as plataformas em y = 650 e y = 700
        # ficavam abaixo da tela (HEIGHT = 600) e nunca apareciam.
        camera_y = player.pos[1] + player.height - (HEIGHT - 50)
        camera_y = max(0, min(camera_y, WORLD_HEIGHT - HEIGHT))

        # A câmera é um vetor de R^2.
        camera = np.array([camera_x, camera_y])

        # Desenho.
        draw_background(screen, camera)
        draw_platforms(screen, camera)
        draw_goal(screen, camera)
        draw_crystals(screen, player, camera)
        draw_vision(screen, enemy, camera, distance, dot, can_see)
        draw_enemy(screen, enemy, camera)
        draw_projectiles(screen, enemy, camera)
        draw_player(screen, player, camera)
        draw_hud(screen, player, enemy, distance, dot, can_see)

        if win_timer > 0:
            win_timer -= dt
            text = title_font.render(win_text, True, GOAL_COLOR)
            screen.blit(text, text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 60)))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
