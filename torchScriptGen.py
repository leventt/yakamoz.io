import os
import torch
from main import ROOT
from importlib.machinery import SourceFileLoader


if __name__ == '__main__':
    surat = SourceFileLoader(
        'surat',
        os.path.join(ROOT, 'surat/surat.py')
    ).load_module()
    surat.DEVICE = torch.device('cpu')

    MODEL_CP_PATH = os.path.join(
        ROOT,
        'checkpoint',
        'checkpoint.pth'
    )

    model = surat.Model(25634)  # TODO
    model.load_state_dict(torch.load(
        MODEL_CP_PATH,
        # for cpu
        map_location=torch.device('cpu')
    ))

    model.eval()

    traced_script_module = torch.jit.trace(
        model,
        (
            torch.zeros((1, 1, 64, 32)),
            torch.zeros(16)
        )
    )

    traced_script_module.save(os.path.join(ROOT, 'yakamoz.pt'))
