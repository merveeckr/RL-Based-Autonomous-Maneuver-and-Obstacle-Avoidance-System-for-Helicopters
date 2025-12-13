import numpy as np
from typing import Dict

class AirSimInterface:
    def __init__(self):
        # Demo/placeholder state (AirSim bağlanana kadar env crash etmesin)
        self._pos = np.zeros(3, dtype=np.float32)
        self._vel = np.zeros(3, dtype=np.float32)
        self._collision = False
        self.domain_params: Dict = {}

    def apply_action(self, action):
        """
        Placeholder: action'a göre çok küçük bir hareket simüle eder.
        AirSim bağlayınca burada gerçek API çağrıları olacak.
        """
        a = np.asarray(action, dtype=np.float32)

        # Örnek: ilk 3 aksiyon x,y,z yönünde küçük bir etki yapsın
        if a.size >= 3:
            self._vel = 0.1 * a[:3]
            self._pos = self._pos + 0.05 * self._vel
        else:
            self._vel[:] = 0.0

    def step(self):
        """
        Placeholder tick. AirSim'de burada sim zamanını ilerletirsin.
        """
        pass

    def get_position(self) -> np.ndarray:
        return self._pos

    def get_velocity(self) -> np.ndarray:
        return self._vel

    def get_collision(self) -> bool:
        return bool(self._collision)

    def apply_domain_randomization(self, params: dict):
        """
        Gün 1 minimum: parametreleri sakla/logla.
        AirSim API destekliyorsa rüzgar/turbulence burada set edilir.
        """
        self.domain_params = dict(params)
