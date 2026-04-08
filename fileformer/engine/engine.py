#  ╔══════════════════════════════════════════════════════════════════╗
#  ║                  Purity Seal of the Omnissiah                    ║
#  ║  ❀  Let this code be sanctified, free from error and heresy.     ║
#  ║  ❀  By the Motive Force, let no null pointer or segfault arise.  ║
#  ║  ❀  The flesh is weak, but the logic endures.                    ║
#  ║  ❀  Praise the Machine God. May the binary be true.              ║
#  ║                                                                  ║
#  ║              (___)                                               ║
#  ║              (o o)   <  Ave Deus Mechanicus!                     ║
#  ║              /"\                                                 ║
#  ║             /_/ \_                                               ║
#  ║                                                                  ║
#  ║                     ~ Puritas Codicis ~                          ║
#  ╚══════════════════════════════════════════════════════════════════╝

from fileformer import FileFormer, ByteLevelTokenizer
import torch
import constriction
import numpy as np
import torch.nn.functional as F
import struct
from .utils import normalize_probabilities

class Engine:
    def __init__(self, seed, model: FileFormer, tokenizer: ByteLevelTokenizer):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()



    def _compress(self, data):
        _tgt = torch.tensor(self.tokenizer.encode(data), dtype=torch.long)
        message_encoder = constriction.stream.queue.RangeEncoder()

        context = torch.tensor([62], dtype=torch.long)

        for i in range(len(_tgt)):
            # Получаем распределение для следующего токена на основе текущего контекста
            with torch.no_grad():
                logits = self.model.forward(context.unsqueeze(0))  # shape (1, seq_len, vocab_size)
                # Берём предсказание для последней позиции (следующий токен)
                probs = normalize_probabilities(logits[0, -1, :], temperature=5.0)
            prob_np = probs.cpu().numpy().astype(np.float32)
            model = constriction.stream.model.Categorical(prob_np, perfect=False)

            sym = _tgt[i].item()  # текущий символ для кодирования
            message_encoder.encode(sym, model)

            # Обновляем контекст: добавляем реальный символ
            context = torch.cat([context, torch.tensor([sym])])
        return message_encoder.get_compressed()

    def _decompress(self, data, len_tgt):
        message_decoder = constriction.stream.queue.RangeDecoder(data)
        context = torch.tensor([62], dtype=torch.long)
        reconstructed = []
        for _ in range(len(len_tgt)):  # декодируем ровно столько символов, сколько было
            with torch.no_grad():
                logits = self.model.forward(context.unsqueeze(0))
                probs = normalize_probabilities(logits[0, -1, :], temperature=5.0)
            prob_np = probs.cpu().numpy().astype(np.float32)
            model = constriction.stream.model.Categorical(prob_np, perfect=False)

            sym = message_decoder.decode(model)
            reconstructed.append(sym)
            context = torch.cat([context, torch.tensor([sym])])

        return self.tokenizer.decode(reconstructed)

    @staticmethod
    def prepare_data_to_save(data):
        header = struct.pack('<I', len(data))
        data = data.astype('<u4').tobytes()
        return header, data

    @staticmethod
    def read_data(data):
        data = np.frombuffer(data, dtype='<u4')
        return data

    def compress(self, data):
        array = self._compress(data)
        return self.prepare_data_to_save(array)

    def decompress(self, data, len_tgt):
        array = self.read_data(data)
        return self._decompress(array, len_tgt)