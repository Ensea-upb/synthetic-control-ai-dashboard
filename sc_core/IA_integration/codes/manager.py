from pathlib import Path
import threading
from typing import Optional

from .status import AIStatus
from .utils_image import matplotlib_to_ov_tensor


class AIManager:
    """
    Gestionnaire du modèle local Qwen2.5-VL via OpenVINO GenAI.

    Supporte :
    - comment_figure(fig, prompt)  → Vision + texte
    - generate_text(prompt)        → Texte seul
    """

    def __init__(self, model_dir: str, device: str = "CPU"):
        self.model_dir = Path(model_dir)
        self.device = device

        self._pipe = None
        self._status = AIStatus.IDLE
        self._error: Optional[Exception] = None
        self._thread: Optional[threading.Thread] = None

    # --------------------------------------------------
    # État
    # --------------------------------------------------

    def status(self) -> AIStatus:
        return self._status

    def is_ready(self) -> bool:
        return self._status == AIStatus.READY

    def error(self) -> Optional[Exception]:
        return self._error

    # --------------------------------------------------
    # Chargement
    # --------------------------------------------------

    def start_background_loading(self):
        if self._status in (AIStatus.LOADING, AIStatus.READY):
            return

        self._status = AIStatus.LOADING
        self._thread = threading.Thread(target=self._load_model, daemon=True)
        self._thread.start()

    def load_blocking(self):
        """Chargement synchrone (utile hors Streamlit)."""
        self._load_model()

    def _load_model(self):
        try:
            import openvino_genai as ov_genai

            if not self.model_dir.exists():
                raise FileNotFoundError(
                    f"Dossier modèle introuvable : {self.model_dir}"
                )

            self._pipe = ov_genai.VLMPipeline(str(self.model_dir), self.device)
            self._status = AIStatus.READY

        except ImportError:
            self._error = ImportError(
                "openvino_genai n'est pas installé. "
                "Installez-le avec : pip install openvino-genai"
            )
            self._status = AIStatus.ERROR

        except Exception as e:
            self._error = e
            self._status = AIStatus.ERROR

    # --------------------------------------------------
    # Inférence — Vision + Texte
    # --------------------------------------------------

    def comment_figure(
        self,
        fig,
        prompt: str,
        max_new_tokens: int = 1000,
    ) -> str:
        """Génère un commentaire à partir d'une figure matplotlib."""
        if not self.is_ready():
            raise RuntimeError(
                f"Modèle non prêt (état={self._status.value}). "
                "Attendez la fin du chargement."
            )

        image_tensor = matplotlib_to_ov_tensor(fig)

        result = self._pipe.generate(
            prompt,
            image=image_tensor,
            max_new_tokens=max_new_tokens,
        )
        return str(result)

    # --------------------------------------------------
    # Inférence — Texte seul
    # --------------------------------------------------

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 1000,
    ) -> str:
        """Génère une réponse texte sans image."""
        if not self.is_ready():
            raise RuntimeError(
                f"Modèle non prêt (état={self._status.value}). "
                "Attendez la fin du chargement."
            )

        result = self._pipe.generate(prompt, max_new_tokens=max_new_tokens)
        return str(result)