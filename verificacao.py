# Verificação das propriedades de Álgebra Linear usadas no jogo.
# Roda sem abrir janela:  python verificacao.py
import math
import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np

import jogo_algebra_linear as jogo

I = np.eye(2)


def perto(a, b):
    return np.allclose(a, b, atol=1e-9)


# 1. Matriz de rotação: ortogonal (R^T R = I), det = 1, preserva a norma
#    e R(theta) E1 é a primeira coluna, (cos theta, sin theta).
for graus in (0, 30, 40, 90, 135, 180, 270):
    t = math.radians(graus)
    R = jogo.rotation_matrix(t)
    v = np.array([3.0, -4.0])
    assert perto(R.T @ R, I)
    assert perto(np.linalg.det(R), 1.0)
    assert perto(np.linalg.norm(R @ v), 5.0)
    assert perto(R @ jogo.E1, [math.cos(t), math.sin(t)])
print("OK rotação: ortogonal, det = 1, preserva a norma")

# 2. Coordenadas homogêneas dão o mesmo resultado que A p + posição,
#    que era a conta do transform_points original.
pos = np.array([560.0, 455.0])
pontos = [(28, 0), (-18, -16), (-18, 16)]
for t in (0.0, 0.7, math.pi):
    for s in (1.0, 1.5):
        A = jogo.rotation_matrix(t) @ jogo.scale_matrix(s, s)
        esperado = [A @ np.array(p, dtype=float) + pos for p in pontos]
        obtido = jogo.apply_matrix(jogo.model_matrix(pos, t, s, s), pontos)
        assert perto(obtido, esperado)
# Com angle = pi o triângulo aponta para a esquerda: a ponta (28, 0) vai para (532, 455).
assert perto(jogo.apply_matrix(jogo.model_matrix(pos, math.pi), [(28, 0)])[0], (532, 455))
print("OK coordenadas homogêneas: T R S p = A p + posição")

# 3. A ordem das matrizes importa: girar depois de transladar é diferente.
T = jogo.translation_matrix(100.0, 0.0)
R3 = jogo.to_homogeneous(jogo.rotation_matrix(math.pi / 2))
assert not perto(T @ R3, R3 @ T)
print("OK composição: T R != R T")

# 4. Reflexão de Householder: H n = -n, H t = t (t perpendicular a n),
#    H^2 = I, det = -1 e |H v| = |v|.
for n in (jogo.E1, jogo.E2, np.array([1.0, 1.0])):
    H = jogo.reflection_matrix(n)
    u = jogo.normalize(n)
    tangente = np.array([-u[1], u[0]])
    v = np.array([200.0, -150.0])
    assert perto(H @ u, -u)
    assert perto(H @ tangente, tangente)
    assert perto(H @ H, I)
    assert perto(np.linalg.det(H), -1.0)
    assert perto(np.linalg.norm(H @ v), np.linalg.norm(v))
print("OK reflexão: H n = -n, H^2 = I, det = -1, preserva a norma")

# 5. Projétil descendo na diagonal sobre o chão (y = 540) volta subindo
#    com a mesma rapidez.
p = jogo.Projectile(np.array([300.0, 520.0]), jogo.normalize(np.array([1.0, 1.0])))
v0 = p.velocity.copy()
for _ in range(10):
    p.update(1 / 60)
assert p.bounces == 1
assert perto(p.velocity, [v0[0], -v0[1]])
print("OK projétil: ricochete no chão inverte só a componente vertical")

# 6. Campo de visão: forward . u >= cos(40°) exatamente quando o ângulo <= 40°.
inimigo = jogo.Enemy()
for graus in (0, 20, 39.9, 40.1, 60, 90, 180):
    u = jogo.rotation_matrix(math.radians(graus)) @ inimigo.forward
    dentro = float(np.dot(inimigo.forward, u)) >= inimigo.fov_threshold
    assert dentro == (graus <= 40)
print("OK campo de visão: produto interno contra cos(40°)")

# 7. Projeção sobre E1 seguida de normalização dá o sentido horizontal.
forward = jogo.normalize(np.array([0.6, -0.8]))
assert perto(jogo.normalize(np.dot(forward, jogo.E1) * jogo.E1), jogo.E1)
print("OK projeção: proj_E1(forward) normalizada = E1")

# 8. Squash and stretch: det(S) = 1 para qualquer velocidade e pouso.
jogador = jogo.Player()
for vy in (0.0, 300.0, -600.0, 1200.0):
    for squash in (0.0, 0.3):
        jogador.velocity[1] = vy
        jogador.squash = squash
        sx, sy = jogador.scale_factors()
        assert perto(np.linalg.det(jogo.scale_matrix(sx, sy)), 1.0)
print("OK squash and stretch: det(S) = 1, área preservada")

# 9. Coleta de cristal por distância.
jogador = jogo.Player()
jogador.pos = jogo.crystals[1] - np.array([jogador.width / 2, jogador.height / 2])
jogador.velocity[:] = 0
jogador.update(0.0, {k: False for k in (jogo.pygame.K_a, jogo.pygame.K_LEFT, jogo.pygame.K_d,
                                        jogo.pygame.K_RIGHT, jogo.pygame.K_w, jogo.pygame.K_SPACE,
                                        jogo.pygame.K_UP)})
assert 1 in jogador.collected
print("OK cristal: coletado quando ||centro - cristal|| <= 30")

print("\nTodas as verificações passaram.")
