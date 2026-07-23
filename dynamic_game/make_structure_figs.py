"""Generate the game-structure timing diagram and attacker surfaces."""
from params import AllParams
import figures

figures.fig_game_structure("outputs/fig_game_structure.png")
figures.fig_attacker_surface(AllParams(), "outputs/fig_attacker_surface.png")
print("ok")
