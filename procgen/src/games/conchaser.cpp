#include "../basic-abstract-game.h"
#include "../assetgen.h"
#include <set>
#include <queue>
#include "../mazegen.h"
#include "../cpp-utils.h"

const std::string NAME = "conchaser";

const float ORB_REWARD = 0.04f;
const float COMPLETION_BONUS = 10.0f;
const float ORB_DIM = 0.3f;

const int LARGE_ORB = 2;
const int ENEMY_WEAK = 3;
const int ENEMY_EGG = 4;
const int MAZE_WALL = 5;
const int ENEMY = 6;
const int ENEMY2 = 7;
const int ENEMY3 = 8;
const int LARGE_ORB_TOXIC = 9;
const int FAKE_LARGE_ORB = 10;
const int FAKE_ORB_TOXIC = 11;

const int MARKER = 1001;
const int ORB = 1002;

class ConfoundedChaserGame : public BasicAbstractGame {
  public:
    std::shared_ptr<MazeGen> maze_gen;
    std::vector<int> free_cells;
    std::vector<bool> is_space_vec;
    int eat_timeout = 0;
    int egg_timeout = 0;
    int eat_time = 0;
    int total_enemies = 0;
    int total_orbs = 0;
    int orbs_collected = 0;
    int maze_dim = 0;
    int always_can_eat = 0;
    int not_allow_respawn = 0;
    int orb_color = 0;

    ConfoundedChaserGame()
        : BasicAbstractGame(NAME) {
        mixrate = 1;
        maxspeed = .5;

        eat_timeout = 75;
        egg_timeout = 50;

        maze_gen = nullptr;
        has_useful_vel_info = false;
    }

    void load_background_images() override {
        main_bg_images_ptr = &topdown_simple_backgrounds;
    }

    void asset_for_type(int type, std::vector<std::string> &names) override {
        if (type == PLAYER) {
            names.push_back("misc_assets/enemyFloating_1b.png");
        } else if (type == ENEMY) {
            names.push_back("misc_assets/enemyFlying_1.png");
        } else if (type == ENEMY2) {
            names.push_back("misc_assets/enemyFlying_2.png");
        } else if (type == ENEMY3) {
            names.push_back("misc_assets/enemyFlying_3.png");
        } else if (type == LARGE_ORB) {
            names.push_back("misc_assets/gemYellow.png");
        } else if (type == LARGE_ORB_TOXIC) {
            names.push_back("misc_assets/gemBlue.png");
        } else if (type == FAKE_LARGE_ORB) {
            names.push_back("misc_assets/gemYellow.png");
        } else if (type == FAKE_ORB_TOXIC) {
            names.push_back("misc_assets/gemBlue.png");
        } else if (type == ENEMY_WEAK) {
            names.push_back("misc_assets/enemyWalking_1b.png");
            //use a diff asset
            // names.push_back("misc_assets/enemyWalking_1.png"); 
        } else if (type == ENEMY_EGG) {
            // names.push_back("misc_assets/enemySpikey_1b.png");
            // Use a diff color enemy egg
            names.push_back("misc_assets/enemySpikey_1.png");
        } else if (type == MAZE_WALL) {
            names.push_back("misc_assets/tileStone_slope.png");
        }
    }

    bool use_block_asset(int type) override {
        return BasicAbstractGame::use_block_asset(type) || (type == MAZE_WALL);
    }

    void update_agent_velocity() override {
        if (action_vx != 0)
            agent->vx = maxspeed * action_vx;
        if (action_vy != 0)
            agent->vy = maxspeed * action_vy;

        // handles some edge cases in collision detection that can reduce velocity
        agent->vx = sign(agent->vx) * maxspeed;
        agent->vy = sign(agent->vy) * maxspeed;
    }

    bool is_blocked(const std::shared_ptr<Entity> &src, int target, bool is_horizontal) override {
        if (target == MAZE_WALL)
            return true;

        return BasicAbstractGame::is_blocked(src, target, is_horizontal);
    }

    int image_for_type(int type) override {
        if (type == ENEMY) {
            // if (can_eat_enemies()) {
                // needs to figure our its weakness status by its speed/locations
                // enemies' shape doesn't change at all, always seems to be "weak"
                return ENEMY_WEAK;
            // } else {
            //     // simulates the enemies walking movements
            //     int rem = (cur_time / 2) % 4;
            //     if (rem == 3)
            //         rem = 1;
            //     return ENEMY + rem;
            // }
        }


        return BasicAbstractGame::image_for_type(type);
    }

    void draw_grid_obj(QPainter &p, const QRectF &rect, int type, int theme) override {
        if (type == ORB) {
            p.fillRect(QRectF(rect.x() + rect.width() * (1 - ORB_DIM) / 2, rect.y() + rect.height() * (1 - ORB_DIM) / 2, rect.width() * ORB_DIM, rect.height() * ORB_DIM), QColor(0, 255, 0));
        } else {
            BasicAbstractGame::draw_grid_obj(p, rect, type, theme);
        }
    }

    void handle_agent_collision(const std::shared_ptr<Entity> &obj) override {
        BasicAbstractGame::handle_agent_collision(obj);

        if (obj->type == LARGE_ORB || obj->type == FAKE_ORB_TOXIC) {
            eat_time = cur_time;
            step_data.reward += ORB_REWARD*5;
            obj->will_erase = true;
        }
        else if (obj->type == LARGE_ORB_TOXIC || obj->type == FAKE_LARGE_ORB) {
            // negative reward for eating toxic ones
            // and there is no invincible buff
            // eat_time = cur_time;
            step_data.reward -= ORB_REWARD*5;
            obj->will_erase = true;
        } else if (obj->type == ENEMY) {
            if (can_eat_enemies()) {
                obj->will_erase = true;
            } else {
                step_data.done = true;
            }
        }
    }

    void choose_world_dim() override {
        main_width = maze_dim;
        main_height = maze_dim;
    }

    void game_reset() override {
        // Hardness Mode setting is replaced by context API altogether
        maze_dim = 11;
        always_can_eat = context.at(0);
        not_allow_respawn = context.at(1);
        total_enemies = context.at(2);
        int extra_orb_sign = context.at(3);
        orb_color = context.at(4);

        if (maze_gen == nullptr) {
            std::shared_ptr<MazeGen> _maze_gen(new MazeGen(&rand_gen, maze_dim));
            maze_gen = _maze_gen;
        }
        BasicAbstractGame::game_reset();

        options.center_agent = false;

        agent->rx = .5;
        agent->ry = .5;

        eat_time = -1 * eat_timeout;

        fill_elem(0, 0, main_width, main_height, MAZE_WALL);

        maze_gen->generate_maze_no_dead_ends();

        free_cells.clear();

        std::vector<std::vector<int>> quadrants;
        std::vector<int> orbs_for_quadrant;
        int num_quadrants = 4;
        int extra_quad = rand_gen.randn(num_quadrants);

        for (int i = 0; i < num_quadrants; i++) {
            std::vector<int> quadrant;
            orbs_for_quadrant.push_back(1 + (i == extra_quad ? extra_orb_sign : 0));
            quadrants.push_back(quadrant);
        }

        for (int i = 0; i < maze_dim; i++) {
            for (int j = 0; j < maze_dim; j++) {
                int obj = maze_gen->grid.get(i + MAZE_OFFSET, j + MAZE_OFFSET);

                set_obj(i, j, obj == WALL_OBJ ? MAZE_WALL : obj);

                if (obj == SPACE) {
                    int idx = j * maze_dim + i;
                    free_cells.push_back(idx);

                    int quad_idx = (i >= maze_dim / 2.0 ? 1 : 0) * 2 + (j >= maze_dim / 2.0 ? 1 : 0);
                    quadrants[quad_idx].push_back(idx);
                }
            }
        }


        for (int i = 0; i < num_quadrants; i++) {
            int num_orbs = orbs_for_quadrant[i];
            std::vector<int> quadrant = quadrants[i];
            std::vector<int> selected_idxs = rand_gen.simple_choose((int)(quadrant.size()), num_orbs);

            for (int j : selected_idxs) {
                int cell = quadrant[j];
                // decide whether to spawn a toxic orb
                // currently toxic orb chance is set at 30% 
                if (rand_gen.randint(0, 100) < 70) {
                    if (orb_color > 0 && orb_color < 4){
                        // all fixed to be yellow
                        spawn_entity_at_idx(cell, 0.4f, LARGE_ORB);
                    } else if (orb_color >= 4){
                        // all fixed to be blue
                        spawn_entity_at_idx(cell, 0.4f, FAKE_ORB_TOXIC);
                    } else {
                        // no change
                        spawn_entity_at_idx(cell, 0.4f, LARGE_ORB);
                    }
                }
                else {
                    if (orb_color > 0 && orb_color < 4){
                        // all fixed to be yellow
                        spawn_entity_at_idx(cell, 0.4f, FAKE_LARGE_ORB);
                    } else if (orb_color >= 4){
                        // all fixed to be blue
                        spawn_entity_at_idx(cell, 0.4f, LARGE_ORB_TOXIC);
                    } else {
                        // no change
                        spawn_entity_at_idx(cell, 0.4f, LARGE_ORB_TOXIC);
                    }
                }
                set_obj(cell, MARKER);
            }
        }


        free_cells = get_cells_with_type(SPACE);
        std::vector<int> selected_idxs = rand_gen.simple_choose((int)(free_cells.size()), 1 + total_enemies);

        int start_idx = selected_idxs[0];
        int start = free_cells[start_idx];

        agent->x = (start % maze_dim) + .5;
        agent->y = (start / maze_dim) + .5;

        for (int i = 0; i < total_enemies; i++) {
            int cell = free_cells[selected_idxs[i + 1]];
            set_obj(cell, MARKER);
            spawn_egg(cell);
        }

        for (int cell : free_cells) {
            set_obj(cell, ORB);
        }

        total_orbs = (int)(free_cells.size());
        orbs_collected = 0;

        std::vector<int> marker_cells = get_cells_with_type(MARKER);

        for (int cell : marker_cells) {
            set_obj(cell, SPACE);
        }


        free_cells.clear();
        is_space_vec.clear();

        for (int i = 0; i < grid_size; i++) {
            bool is_space = get_obj(i) != MAZE_WALL;

            if (is_space) {
                free_cells.push_back(i);
            }

            is_space_vec.push_back(is_space);
        }
    }

    bool can_eat_enemies() {
        return (cur_time - eat_time < eat_timeout) || always_can_eat > 0;
    }

    void spawn_egg(int enemy_cell) {
        auto egg = add_entity((enemy_cell % maze_dim) + .5, (enemy_cell / maze_dim) + .5, 0, 0, .5, ENEMY_EGG);
        egg->health = egg_timeout;
    }

    int manhattan_dist(int a, int b) {
        return abs((a % main_width) - (b % main_width)) + abs((a / main_width) - (b / main_width));
    }

    void get_adjacent(int idx, std::vector<int> &neighbors) {
        int x = idx % main_width;
        int y = idx / main_width;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (i == 0 && j == 0)
                    continue;
                if (i != 0 && j != 0)
                    continue;

                int neighbor = to_grid_idx(x + i, y + j);

                if (neighbor != INVALID_IDX) {
                    neighbors.push_back(neighbor);
                }
            }
        }
    }

    void game_step() override {
        // update all moving objs locations and call handle collisions
        BasicAbstractGame::game_step();

        int num_orbs = 0;
        int num_enemies = 0;

        float default_enemy_speed = .5;
        // make the weakened enemies even slower
        float vscale = (can_eat_enemies()) ? (default_enemy_speed * .5) : (default_enemy_speed); 
        step_data.can_eat = can_eat_enemies();

        for (int j = (int)(entities.size()) - 1; j >= 0; j--) {
            auto ent = entities[j];

            if (ent->type == ORB) {
                num_orbs++;
            } else if (ent->type == ENEMY_EGG) {
                num_enemies++;
                ent->health -= 1;

                if (ent->health == 0) {
                    ent->will_erase = true;
                    auto enemy = spawn_child(ent, ENEMY, .5);
                    enemy->smart_step = true;
                }
            } else if (ent->type == ENEMY) {
                num_enemies++;

                float x = ent->x - .5;
                float y = ent->y - .5;

                int dist_scale = (can_eat_enemies()) ? -1 : 1;

                int enemy_idx = to_grid_idx(x, y);
                int agent_idx = to_grid_idx(agent->x, agent->y);

                bool is_at_junction = fabs(x - round(x)) + fabs(y - round(y)) < .01;
                bool be_agressive = step_rand_int % 2 == 0;

                if ((ent->vx == 0 && ent->vy == 0) || is_at_junction) {
                    std::vector<int> adj_elems;
                    std::vector<int> space_neighbors;
                    int prev_idx = to_grid_idx(x - sign(ent->vx), y - sign(ent->vy));
                    get_adjacent(enemy_idx, adj_elems);

                    int min_dist = 2 * main_width;

                    for (int adj : adj_elems) {
                        if (is_space_vec[adj] && adj != prev_idx) {
                            int md = manhattan_dist(adj, agent_idx) * dist_scale;

                            if (be_agressive) {
                                if (md < min_dist) {
                                    min_dist = md;
                                    space_neighbors.clear();
                                    space_neighbors.push_back(adj);
                                } else if (md == min_dist) {
                                    space_neighbors.push_back(adj);
                                }
                            } else {
                                space_neighbors.push_back(adj);
                            }
                        }
                    }

                    int neighbor_idx = step_rand_int % space_neighbors.size();
                    int neighbor = space_neighbors[neighbor_idx];

                    int nx = neighbor % main_width;
                    int ny = neighbor / main_width;

                    ent->vx = (nx - x) * vscale;
                    ent->vy = (ny - y) * vscale;
                }
            }
        };

        // For causally not aligned curriculum, they will set not-allow-respawn = 1
        if (num_enemies < total_enemies && not_allow_respawn == 0) {
            int selected_idx = step_rand_int % free_cells.size();
            spawn_egg(free_cells[selected_idx]);
        }

        int agent_idx = get_agent_index();

        if (get_obj(agent_idx) == ORB) {
            set_obj(agent_idx, SPACE);
            step_data.reward += ORB_REWARD;
            orbs_collected += 1;
        }

        if (orbs_collected == total_orbs) {
            step_data.reward += COMPLETION_BONUS;
            step_data.level_complete = true;
            step_data.done = true;
        }
    }

    void serialize(WriteBuffer *b) override {
        BasicAbstractGame::serialize(b);
        b->write_vector_int(free_cells);
        b->write_vector_bool(is_space_vec);
        b->write_int(eat_timeout);
        b->write_int(egg_timeout);
        b->write_int(eat_time);
        b->write_int(total_enemies);
        b->write_int(total_orbs);
        b->write_int(orbs_collected);
        b->write_int(maze_dim);
    }

    void deserialize(ReadBuffer *b) override {
        BasicAbstractGame::deserialize(b);
        free_cells = b->read_vector_int();
        is_space_vec = b->read_vector_bool();
        eat_timeout = b->read_int();
        egg_timeout = b->read_int();
        eat_time = b->read_int();
        total_enemies = b->read_int();
        total_orbs = b->read_int();
        orbs_collected = b->read_int();
        maze_dim = b->read_int();
    }
};

REGISTER_GAME(NAME, ConfoundedChaserGame);
