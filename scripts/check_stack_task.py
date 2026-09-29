"""Run the fixed-layout stacking trajectory without creating a renderer."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from stack_color_blocks import StackColorBlocksEnv
from run_scene import COLORS, run_stack
from build_scene import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pick-color', choices=COLORS, required=True)
    parser.add_argument('--place-color', choices=COLORS, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=90000)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/task_check')
    parser.add_argument('--random-blocks', action='store_true',
                        help='replay the seeded randomized layout instead of fixed positions')
    args = parser.parse_args()
    if args.pick_color == args.place_color:
        parser.error('--pick-color and --place-color must be different')
    if args.max_steps < 1:
        parser.error('--max-steps must be positive')
    args.task = 'color_block_stack'
    args.fixed_blocks = not args.random_blocks
    args.randomize_background = False
    args.randomize_lighting = False
    args.headless = True
    args.output.mkdir(parents=True, exist_ok=True)
    build()
    env = StackColorBlocksEnv(render=True, headless=True, render_hz=0)
    try:
        run_stack(env, args, args.output)
    finally:
        env.close()


if __name__ == '__main__':
    main()
