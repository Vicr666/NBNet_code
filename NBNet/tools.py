import torch


def look_state(path):
    device_id = torch.cuda.current_device()
    resume_state = torch.load(path, map_location=lambda storage, loc: storage.cuda(device_id))
    print(resume_state)


if __name__ == '__main__':
    look_state('/root/autodl-tmp/experiments/185000.state')