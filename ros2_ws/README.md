# TSB Recruitment 2026: Turtlesim + Behavior Trees + PyTorch



## Final job
Your job is to make the turtlesim turtle move on its own. A **PyTorch** network
looks at images and decides where to go. A **Behavior Tree**, which you edit in
**Groot2**, turns that decision into motion.

This repository gives you the architecture, build setup and skeleton files.
**You write the logic.**

---

## 1. Final Architecture

```
               sensor_msgs/Image                    std_msgs/String
 image source ──────────────────▶ ┌──────────────┐ ──────────────────▶ ┌──────────────────────────┐
 (/vision/image)                  │ vision_node  │  /vision/command    │ bt_executor              │
                                  │ tsb_cv_brain │  LEFT|RIGHT|FORWARD │ tsb_bt_core              │
                                  │  (PyTorch)   │                     │                          │
                                  └──────────────┘                     │  Sequence                │
                                                                       │   ├─ GetVisionCommand    │
                                                                       │   └─ MoveTurtle ─────────┼──▶ /turtle1/cmd_vel ──▶ turtlesim
                                                                       └───────────▲──────────────┘   geometry_msgs/Twist
                                                                                   │ ZMQ :1667
                                                                                 Groot2
```

| Package | Build type | Language | Responsibility |
|---|---|---|---|
| `tsb_bt_core` | `ament_cmake` | C++ / XML | Loads and ticks the Behavior Tree (BehaviorTree.CPP **v4**), sends velocity commands to turtlesim |
| `tsb_cv_brain` | `ament_python` | Python | Runs the PyTorch network on incoming images and publishes the command |

### Interface between the two packages

| Topic | Type | Publisher | Subscriber | Values |
|---|---|---|---|---|
| `/vision/image` | `sensor_msgs/msg/Image` | image source | `vision_node` | raw images |
| `/vision/command` | `std_msgs/msg/String` | `vision_node` | `GetVisionCommand` (BT) | `LEFT`, `RIGHT`, `FORWARD` |
| `/turtle1/cmd_vel` | `geometry_msgs/msg/Twist` | `MoveTurtle` (BT) | `turtlesim` | velocity |

You may change this interface, but keep both sides in sync.

### Workspace layout

```
recrutamento2026/                      <- workspace root (run colcon here)
├── README.md
└── src/
    ├── tsb_bt_core/
    │   ├── CMakeLists.txt
    │   ├── package.xml
    │   ├── src/bt_executor.cpp                          <- node + factory + tick loop
    │   ├── include/tsb_bt_core/actions/
    │   │   ├── get_vision_command.hpp                   <- BT Action (TODO)
    │   │   └── move_turtle.hpp                          <- BT Action (TODO)
    │   ├── behavior_trees/
    │   │   ├── tree.xml                                 <- the Behavior Tree (edit in Groot2)
    │   │   └── tsb_challenge.btproj                     <- Groot2 project
    │   └── launch/challenge.launch.py                   <- starts everything
    └── tsb_cv_brain/
        ├── package.xml
        ├── setup.py / setup.cfg
        ├── requirements.txt                             <- pip dependencies (PyTorch)
        ├── models/                                      <- put trained weights (.pt/.pth) here
        └── tsb_cv_brain/vision_node.py                  <- CNN + ROS node (TODO)
```

---

## 2. Prerequisites

* Ubuntu 22.04 + **ROS 2 Humble**(recommended), or Ubuntu 24.04 + **ROS 2 Jazzy**
* BehaviorTree.CPP v4 and turtlesim:

  ```bash
  sudo apt install ros-$ROS_DISTRO-behaviortree-cpp ros-$ROS_DISTRO-turtlesim
  ```

  > On Humble, use `ros-humble-behaviortree-cpp` (v4). **Do not** use
  > `ros-humble-behaviortree-cpp-v3`.

* The other ROS dependencies, installed from the workspace root:

  ```bash
  sudo rosdep init   # only if you have never run it
  rosdep update
  rosdep install --from-paths src --ignore-src -y
  ```

* **PyTorch.** It is not a rosdep key, so install it with pip:

  ```bash
  pip install -r src/tsb_cv_brain/requirements.txt
  ```

  * CPU only (smaller download):
    `pip install torch --index-url https://download.pytorch.org/whl/cpu`
  * Ubuntu 24.04 / Jazzy blocks system-wide pip installs (PEP 668). Either add
    `--break-system-packages`, or create a venv that can still see ROS:
    `python3 -m venv --system-site-packages ~/tsb_venv && source ~/tsb_venv/bin/activate`.
  * Check the install: `python3 -c "import torch; print(torch.__version__)"`

* **Groot2**: download it from <https://www.behaviortree.dev/groot>.

---

## 3. Build

Run these from the workspace root (`recrutamento2026/`):

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
colcon build --symlink-install
source install/setup.bash
```

`--symlink-install` means:
* Edits to Python files take effect without rebuilding.
* Edits to `tree.xml`, including edits made in Groot2, take effect without
  rebuilding. Just restart `bt_executor`.
* **C++ changes always need a new `colcon build`.**

To rebuild only one package: `colcon build --symlink-install --packages-select tsb_bt_core`

---

## 4. Run

### Everything at once (turtlesim + vision node + BT executor)

```bash
ros2 launch tsb_bt_core challenge.launch.py
```

Optional arguments:

```bash
ros2 launch tsb_bt_core challenge.launch.py \
    model_path:=$PWD/src/tsb_cv_brain/models/my_model.pt \
    tree_file:=$PWD/src/tsb_bt_core/behavior_trees/tree.xml \
    groot_port:=1667
```

### Or node by node, one terminal each (remember to `source install/setup.bash`)

```bash
ros2 run turtlesim turtlesim_node
ros2 run tsb_cv_brain vision_node --ros-args -p model_path:=/path/to/model.pt
ros2 run tsb_bt_core bt_executor
```

`bt_executor` parameters:

| Parameter | Default | Description |
|---|---|---|
| `tree_file` | `<share>/tsb_bt_core/behavior_trees/tree.xml` | Behavior Tree to load |
| `tick_rate_hz` | `10.0` | How often the tree is re-ticked |
| `groot_port` | `1667` | Port for Groot2 live monitoring |
| `print_tree_nodes_model` | `false` | Print the `<TreeNodesModel>` XML for Groot2 |

`vision_node` parameters: `image_topic` (`/vision/image`), `command_topic`
(`/vision/command`), `model_path` (`""`).

### Test the BT without the neural network

You can pretend to be the vision node and publish a command by hand:

```bash
ros2 topic pub --once /vision/command std_msgs/msg/String "{data: LEFT}"
ros2 topic echo /turtle1/cmd_vel
```

### Out of the box

Before you implement anything, the whole system already builds and launches.
The placeholder nodes log `... is not implemented yet` and return `FAILURE`,
so the turtle does not move.

---

## 5. Groot2

1. Open Groot2, then **Load project** →
   `src/tsb_bt_core/behavior_trees/tsb_challenge.btproj`.
2. Edit `MainTree` in `tree.xml`. For example, connect the `command` output of
   `GetVisionCommand` to the `command` input of `MoveTurtle` through a
   blackboard entry such as `{vision_command}`. Save the file (Ctrl+S).
3. Restart `bt_executor`. No rebuild is needed thanks to `--symlink-install`.
4. **Live monitoring**: with `bt_executor` running, open the **Monitor** tab in
   Groot2 and connect to `localhost`, port `1667`.

If you add or change ports in C++, update the `<TreeNodesModel>` sections in
`tree.xml` and `tsb_challenge.btproj`. You can generate the model with:

```bash
ros2 run tsb_bt_core bt_executor --ros-args -p print_tree_nodes_model:=true
```

---

## 6. What you need to implement

Search for `TODO (candidate)`:

```bash
grep -rn "TODO (candidate)" src/
```

| File | What to do |
|---|---|
| `src/tsb_cv_brain/tsb_cv_brain/vision_node.py` → `TurtleVisionCNN.__init__` / `forward` | Design the network: `// TODO: define the network layers` / `// TODO: implement the forward pass` |
| `src/tsb_cv_brain/tsb_cv_brain/vision_node.py` → `VisionNode.__init__` | `// TODO: Implement PyTorch model loading here` (weights, `eval()`, error handling) |
| `src/tsb_cv_brain/tsb_cv_brain/vision_node.py` → `image_callback`, `preprocess`, `postprocess` | Image → tensor → inference → `LEFT`/`RIGHT`/`FORWARD` → `publish_command()` |
| `src/tsb_bt_core/include/tsb_bt_core/actions/get_vision_command.hpp` → `tick()` | `// TODO: read the latest command, write the "command" output port, return a status` |
| `src/tsb_bt_core/include/tsb_bt_core/actions/move_turtle.hpp` → `tick()` | `// TODO: read the "command" input port, build a Twist and publish it` |
| `src/tsb_bt_core/src/bt_executor.cpp` | `// TODO: register extra nodes` / `// TODO: decide what happens when the tree finishes` |
| `src/tsb_bt_core/behavior_trees/tree.xml` | `<!-- TODO: configure the ports / structure of the tree -->` (preferably in Groot2) |
| `src/tsb_bt_core/CMakeLists.txt` | `# TODO: add new .cpp files` if you split the code |

**Already provided (you don't need to write these):**
* the ROS publishers and subscribers
* the thread-safe storage of the last received command
* the executor thread
* tree loading and Groot2 monitoring
* parameters and the launch file

### Tips

* In BT.CPP v4, `getInput<T>()` returns a `BT::Expected<T>`. Check it before
  using the value.
* A node that must *wait* or *move for some time* should be a
  `BT::StatefulActionNode` that returns `RUNNING`, not a `SyncActionNode`.
  Remember to stop the turtle in `onHalted()`.
* Turtlesim coordinates run from 0 to about 11 on both axes. `ros2 service call /reset std_srvs/srv/Empty`
  puts the turtle back at the start.
* Put trained weights in `src/tsb_cv_brain/models/`. Run `colcon build` again
  after adding a new file so it gets installed to
  `share/tsb_cv_brain/models`.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not find a package configuration file provided by "behaviortree_cpp"` | `sudo apt install ros-$ROS_DISTRO-behaviortree-cpp`, then open a new terminal and `source /opt/ros/$ROS_DISTRO/setup.bash` |
| `ModuleNotFoundError: No module named 'torch'` | Install PyTorch into the **same** `python3` that ROS uses (section 2) |
| `Package 'tsb_...' not found` | Run `source install/setup.bash` in every new terminal |
| Groot2 shows the nodes in red or as unknown | `<TreeNodesModel>` does not match the C++ `providedPorts()` (section 5) |
| Groot2 Monitor does not connect | Check that `bt_executor` is running and the port matches `groot_port` |
