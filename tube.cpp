/*
 *  This is the performance implementation of TUBE model.
 */

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <random>
#include <fstream>
#include <set>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <thread>
#include <numeric>
#include <cmath>

using namespace std;

/* Precisions */
typedef long int lint;
typedef double real;

/* Goals, Contexts, Plans and mappings */
lint num_goals, num_contexts, num_plans, num_behaviors;
vector<string> goals_, contexts_;
map<string, lint> goal2goal_x, context2context_x;
vector<vector<lint>> plans_;
map<vector<lint>, lint> context_xs_sorted2plan_x;
vector<vector<lint>> goal_x_plan_x_prs_;

/* Embeddings */
vector<vector<real>> goal_x2emb_vec;
vector<vector<real>> context_x2emb_vec;
// Optimal embeddings with min training loss
vector<vector<real>> optimal_goal_x2emb_vec;
vector<vector<real>> optimal_context_x2emb_vec;

/* Parameters */
// Required
string behaviors_file, goal_embs_file, context_embs_file;
// Optional
lint emb_dim_num  = 8, threads_num = 2, total_samples_num = 0,
    negative_samples_num = 1;
real init_learning_rate = 0.005;

/* Misc */
lint total_samples_unit = (lint)1e6;
real total_samples_input = 0;

lint learning_rate_mode = 2;  // 1 for constant; 2 for linearly decay (default); 3 for exponential decay


/*
 * Compute \phi(p|g) according to Eqn.~(2)
 */
real get_phi(real& pg, real& gg) {
    return (pg - gg) / sqrt(gg);
}


/*
 * Compute \eps(p|g) according to Eqn.~(3)
 */
real get_eps(real& pp, real& pg, real& gg) {
    auto phi = get_phi(pg, gg);
    if (phi >= 0) {
        return sqrt((pp * gg - pg * pg) / gg);
    } else {
        return sqrt(pp + gg - 2 * pg);
    }
}


real _easy_tanh(real x) {
    // If needed
    return tanh(x);
}


/*
 * Compute positive loss according to Eqn.~(7)
 */
real get_pos_loss(real& pos_eps, int optimization=1) {
    switch (optimization) {
        case 1:
            return - log2(_easy_tanh(1 / (2 * pow(pos_eps, 2))));
        default:
            return - log2(tanh(1 / (2 * pow(pos_eps, 2))));
    }
}


/*
 * Compute negative loss according to Eqn.~(19)
 */
real get_neg_loss(real& neg_eps, int optimization=1) {
    switch (optimization) {
        case 1:
            return - log2(_easy_tanh(pow(neg_eps, 2) / 2));
        default:
            return - log2(tanh(pow(neg_eps, 2) / 2));
    }
}


real _easy_sinh(real x, real high=15.0, real low=1e-5) {
    if (x > high) {
        return sinh(high);
    } else if (x < low) {
        return sinh(low);
    } else {
        return sinh(x);
    }
}


/*
 * Thread for training
 */
void *train_thread(void* thread_id) {
    auto th_id = (long)thread_id;

    /* Parameters */
    // Equally distribute samples to threads
    auto thread_samples_num = total_samples_num / threads_num + 1;
    // How often to print progress information
    auto checkpoints_interval = (lint)(num_behaviors / threads_num + 1);  // Each round
    // Set initial learning rate
    real curr_learning_rate = init_learning_rate;
    // Vector of total loss, pos loss, and neg loss per checkpoint
    vector<vector<real>> checkpoints_losses_;
    // Random engine
    random_device rd;
    mt19937 engine(rd());
    // Misc
    real checkpoint_pos_loss = 0, checkpoint_neg_loss = 0;
    real optimal_total_loss = 0;
    lint optimal_checkpoint = 0;

    for (lint curr_samples_num = 1; curr_samples_num <= thread_samples_num; curr_samples_num ++) {
        /*
         * Test for early drop conditions
         * */
        if (checkpoints_losses_.size() >= 2) {
            // If new checkpoint total loss is more than 150% of initial total loss.
            if (checkpoints_losses_.back()[0] / checkpoints_losses_.front()[0] > 1.50) {
                if (th_id == 0) {
                    cout << "Training wrong! Drop @" << setw(3) << checkpoints_losses_.size() << endl;
                    cout << "Use optimal embeddings saved @" << setw(3) << optimal_checkpoint << endl;
                }
                break;
            }
        }

        /*
         * Positive sample
         * */
        // Sample a positive action index
        uniform_int_distribution<lint> dist_behaviors_size(0, goal_x_plan_x_prs_.size() - 1);
        auto pos_behavior_x = dist_behaviors_size(engine);
        auto pos_goal_x = goal_x_plan_x_prs_[pos_behavior_x][0];
        auto pos_plan_x = goal_x_plan_x_prs_[pos_behavior_x][1];
        // Get Goal vector and compute Plan vector in positive action
        auto pos_goal_vec = goal_x2emb_vec[pos_goal_x];
        vector<real> pos_plan_vec(static_cast<unsigned long>(emb_dim_num));
        for (const auto& context_x: plans_[pos_plan_x]) {
            for(lint d = 0; d < emb_dim_num; d ++) {
                pos_plan_vec[d] += context_x2emb_vec[context_x][d];
            }
        }
        // Compute Phi, Eps, loss, and etc.
        real _pos_pp = inner_product(pos_plan_vec.begin(), pos_plan_vec.end(), pos_plan_vec.begin(), 0.0);
        real _pos_pg = inner_product(pos_plan_vec.begin(), pos_plan_vec.end(), pos_goal_vec.begin(), 0.0);
        real _pos_gg = inner_product(pos_goal_vec.begin(), pos_goal_vec.end(), pos_goal_vec.begin(), 0.0);
        auto _pos_phi = get_phi(_pos_pg, _pos_gg);
        auto _pos_eps = get_eps(_pos_pp, _pos_pg, _pos_gg);
        auto _pos_loss = get_pos_loss(_pos_eps);
        checkpoint_pos_loss += _pos_loss;
        auto _pos_grad_deno_1 =
                _easy_sinh(_pos_gg/(_pos_pp*_pos_gg-_pos_pg*_pos_pg)) * pow((_pos_pp*_pos_gg-_pos_pg*_pos_pg), 2);
        auto _pos_grad_deno_2 = _easy_sinh(1/(_pos_pp+_pos_gg-2*_pos_pg)) * pow((_pos_pp+_pos_gg-2*_pos_pg), 2);

        // Compute gradient vec w.r.t. Context (Eqn.~(14)), and update Context emb
        vector<real> pos_context_grad_vec (static_cast<unsigned long>(emb_dim_num));
        for(lint d = 0; d < emb_dim_num; d ++) {
            if (_pos_phi >= 0.0) {
                pos_context_grad_vec[d] += (2 * _pos_pg * _pos_gg / _pos_grad_deno_1) * pos_goal_vec[d];
                pos_context_grad_vec[d] -= (2 * _pos_gg * _pos_gg / _pos_grad_deno_1) * pos_plan_vec[d];
            } else {
                pos_context_grad_vec[d] += (2 / _pos_grad_deno_2) * pos_goal_vec[d];
                pos_context_grad_vec[d] -= (2 / _pos_grad_deno_2) * pos_plan_vec[d];
            }
        }
        for (const auto& context_x: plans_[pos_plan_x]) {
            for(lint d = 0; d < emb_dim_num; d ++) {
                context_x2emb_vec[context_x][d] += (pos_context_grad_vec[d] * curr_learning_rate);
            }
        }

        // Compute gradient vec w.r.t. Goal (Eqn.~(18)), and update Goal emb
        vector<real> pos_goal_grad_vec (static_cast<unsigned long>(emb_dim_num));
        for(lint d = 0; d < emb_dim_num; d ++) {
            if (_pos_phi >= 0.0) {
                pos_goal_grad_vec[d] += (2 * _pos_pg * _pos_gg / _pos_grad_deno_1) * pos_plan_vec[d];
                pos_goal_grad_vec[d] -= (2 * _pos_pg * _pos_pg / _pos_grad_deno_1) * pos_goal_vec[d];
            } else {
                pos_goal_grad_vec[d] += (2 / _pos_grad_deno_2) * pos_plan_vec[d];
                pos_goal_grad_vec[d] -= (2 / _pos_grad_deno_2) * pos_goal_vec[d];
            }
        }
        for(lint d = 0; d < emb_dim_num; d ++) {
            goal_x2emb_vec[pos_goal_x][d] += (pos_goal_grad_vec[d] * curr_learning_rate);
        }

        /*
         * Negative samples
         * */
        vector<lint> neg_goal_xs_;
        vector<vector<lint>> neg_plan_context_xs_;
        vector<vector<real>> neg_goal_vecs_, neg_plan_vecs_;
        // Build negative behaviors
        for (lint n = 0; n < negative_samples_num; n ++) {
            // Keep plan fixed (as positive plan), and randomly sample a different goal.
            auto neg_goal_x = pos_goal_x;
            uniform_int_distribution<lint> dist_goals_size(0, goals_.size() - 1);
            while (neg_goal_x == pos_goal_x) {
                neg_goal_x = dist_goals_size(engine);
            }
            auto neg_goal_vec = goal_x2emb_vec[neg_goal_x];
            // New Goal
            neg_goal_xs_.push_back(neg_goal_x);
            neg_goal_vecs_.push_back(neg_goal_vec);
            // Fixed Plan
            neg_plan_context_xs_.push_back(plans_[pos_plan_x]);
            neg_plan_vecs_.push_back(pos_plan_vec);

            // Randomly take a subset (size of 1) of positive plan as negative plan, same goal as positive goal
            if (plans_[pos_plan_x].size() > 1) {  // Skip when positive plan only contains a single context
                vector<lint> subset_pos_plan_context_xs_;
                vector<real> neg_plan_vec(static_cast<unsigned long>(emb_dim_num));

                uniform_int_distribution<lint> dist_pos_plan_size(0, plans_[pos_plan_x].size() - 1);
                auto _sampled_context_x = plans_[pos_plan_x][dist_pos_plan_size(engine)];
                subset_pos_plan_context_xs_.push_back(_sampled_context_x);
                neg_plan_vec = context_x2emb_vec[_sampled_context_x];

                // Fixed Goal
                neg_goal_xs_.push_back(pos_goal_x);
                neg_goal_vecs_.push_back(pos_goal_vec);
                // Subset of positive Plan
                neg_plan_context_xs_.push_back(subset_pos_plan_context_xs_);
                neg_plan_vecs_.push_back(neg_plan_vec);
            }
        }
        // For each negative sample, compute gradient and update embeddings
        for (unsigned int n = 0; n < neg_goal_xs_.size(); n ++) {
            real _neg_pp = inner_product(neg_plan_vecs_[n].begin(), neg_plan_vecs_[n].end(), neg_plan_vecs_[n].begin(), 0.0);
            real _neg_pg = inner_product(neg_plan_vecs_[n].begin(), neg_plan_vecs_[n].end(), neg_goal_vecs_[n].begin(), 0.0);
            real _neg_gg = inner_product(neg_goal_vecs_[n].begin(), neg_goal_vecs_[n].end(), neg_goal_vecs_[n].begin(), 0.0);

            auto _neg_phi = get_phi(_neg_pg, _neg_gg);
            auto _neg_eps = get_eps(_neg_pp, _neg_pg, _neg_gg);
            auto _neg_loss = get_neg_loss(_neg_eps);
            checkpoint_neg_loss += _neg_loss;

            auto _neg_grad_deno_1 = _easy_sinh((_neg_pp * _neg_gg - _neg_pg * _neg_pg) / _neg_gg);
            auto _neg_grad_deno_2 = _easy_sinh(_neg_pp + _neg_gg - 2 * _neg_pg);

            // Compute gradient vec w.r.t. Context (Eqn.~(21)), and update Context emb
            vector<real> neg_context_grad_vec (static_cast<unsigned long>(emb_dim_num));
            for(lint d = 0; d < emb_dim_num; d ++) {
                if (_neg_phi >= 0.0) {
                    neg_context_grad_vec[d] += ((2 * _neg_gg) / (_neg_grad_deno_1 * _neg_gg)) * neg_plan_vecs_[n][d];
                    neg_context_grad_vec[d] -= ((2 * _neg_pg) / (_neg_grad_deno_1 * _neg_gg)) * neg_goal_vecs_[n][d];
                } else {
                    neg_context_grad_vec[d] += (2 / _neg_grad_deno_2) * neg_plan_vecs_[n][d];
                    neg_context_grad_vec[d] -= (2 / _neg_grad_deno_2) * neg_goal_vecs_[n][d];
                }
            }
            for (const auto& context_x: neg_plan_context_xs_[n]) {
                for(lint d = 0; d < emb_dim_num; d ++) {
                    context_x2emb_vec[context_x][d] += (neg_context_grad_vec[d] * curr_learning_rate);
                }
            }

            // Compute gradient vec w.r.t. Goal (Eqn.~(22)), and update Goal emb
            vector<real> neg_goal_grad_vec (static_cast<unsigned long>(emb_dim_num));
            for(lint d = 0; d < emb_dim_num; d ++) {
                if (_neg_phi >= 0.0) {
                    neg_goal_grad_vec[d] += ((2 * _neg_pg * _neg_pg) / (_neg_grad_deno_1 * _neg_gg * _neg_gg)) * neg_goal_vecs_[n][d];
                    neg_goal_grad_vec[d] -= ((2 * _neg_pg * _neg_gg) / (_neg_grad_deno_1 * _neg_gg * _neg_gg)) * neg_plan_vecs_[n][d];
                } else {
                    neg_goal_grad_vec[d] += (2 / _neg_grad_deno_2) * neg_goal_vecs_[n][d];
                    neg_goal_grad_vec[d] -= (2 / _neg_grad_deno_2) * neg_plan_vecs_[n][d];
                }
            }
            for(lint d = 0; d < emb_dim_num; d ++) {
                goal_x2emb_vec[neg_goal_xs_[n]][d] += (neg_goal_grad_vec[d] * curr_learning_rate);
            }
        }


        /*
         * Check for checkpoints: update optimal Goal, Context embeddings if necessary; update learning rate
         */
        // Check for checkpoints
        if (curr_samples_num >= checkpoints_interval && curr_samples_num % checkpoints_interval == 0) {
            real _nor_cp_pos_loss = checkpoint_pos_loss * threads_num;
            real _nor_cp_neg_loss = (checkpoint_neg_loss / (2 * negative_samples_num)) * threads_num;
            real _nor_cp_total_loss = _nor_cp_pos_loss + _nor_cp_neg_loss;
            vector<real> checkpoint_losses_ {_nor_cp_total_loss, _nor_cp_pos_loss, _nor_cp_neg_loss};
            checkpoints_losses_.push_back(checkpoint_losses_);
            checkpoint_pos_loss = 0;
            checkpoint_neg_loss = 0;

            // Update optimal Goal, Context embeddings
            if (checkpoints_losses_.size() == 1){
                optimal_total_loss = checkpoints_losses_.front()[0];
                optimal_checkpoint = 1;
            } else if (checkpoints_losses_.size() >= 2 && checkpoints_losses_.back()[0] < optimal_total_loss) {
                optimal_total_loss = checkpoints_losses_.back()[0];
                optimal_checkpoint = checkpoints_losses_.size();

                optimal_goal_x2emb_vec = goal_x2emb_vec;
                optimal_context_x2emb_vec = context_x2emb_vec;
            }

            /* Update learning rate */
            if (learning_rate_mode == 2) {  // Linear decay
                real _ratio = 1.0 - (real) curr_samples_num / thread_samples_num;
                curr_learning_rate = init_learning_rate * _ratio;
            } else if (learning_rate_mode == 3) {  // Exponential decay
                real _checkpoints = checkpoints_losses_.size();
                curr_learning_rate = init_learning_rate * exp(-(0.01 * _checkpoints));
            }
        }

        /*
         * Report progress
         * */
        if (th_id == 0) {
            if (curr_samples_num == 1) {
                cout << setw(9)  << "Progress" << setw(18)  << "Total loss" << setw(23)  << "Positive loss"
                     << setw(23)  << "Negative loss" << endl;
            }
            if (curr_samples_num >= checkpoints_interval && curr_samples_num % checkpoints_interval == 0) {
                real _total_loss_per = checkpoints_losses_.back()[0] / checkpoints_losses_.front()[0] * 100;
                real _pos_loss_per = checkpoints_losses_.back()[1] / checkpoints_losses_.front()[1] * 100;
                real _neg_loss_per = checkpoints_losses_.back()[2] / checkpoints_losses_.front()[2] * 100;
                cout.precision(2);
                cout << fixed << "Loss @" << setw(3) << checkpoints_losses_.size() << ":"
                     << setw(11) << checkpoints_losses_.back()[0] << setw(8) << _total_loss_per << "%  |"
                     << setw(11) << checkpoints_losses_.back()[1] << setw(8) << _pos_loss_per << "%  |"
                     << setw(11) << checkpoints_losses_.back()[2] << setw(8) << _neg_loss_per << "%  |"
                     << endl;
            }
            if (curr_samples_num == thread_samples_num) {
                cout << "Use optimal embeddings saved @" << setw(3) << optimal_checkpoint << endl;
            }
        }
    }
    pthread_exit(nullptr);
}


/*
 * Training by multi-threading
 */
void train() {
    long thread_id;
    pthread_t threads[threads_num];

    cout << "Start training ..." << endl;
    for (thread_id = 0; thread_id < threads_num; thread_id ++) {
        pthread_create(&threads[thread_id], nullptr, train_thread, (void *)thread_id);
        //pthread_create(&threads[thread_id], nullptr, train_learn_suc_thread, nullptr);
    }
    for (thread_id=0; thread_id<threads_num; ++thread_id) {
        pthread_join(threads[thread_id], nullptr);
    }
    cout << "Done!" << endl;
}


/*
 * Read in behaviors information from external file.
 * Each line follows format: <goal_str>\t<context_str_1>[,<context_str_2>,...]
 */
void read_behaviors_file(const string& behaviors_file, char delim_l1='\t', char delim_l2=',') {
    cout << "Reading behaviors file " << behaviors_file << " ..." << endl;
    ifstream filein(behaviors_file);
    if (!filein) {
        cout << "Error: behaviors file not found!" << endl;
        exit(1);
    }

    set<string> _goals_set, _contexts_set, _plans_set;
    for (string line; getline(filein, line); ) {
        vector<string> tokens;
        stringstream ss(line);
        string token;
        while (getline(ss, token, delim_l1)) {
            tokens.push_back(token);
        }
        if(tokens.size() != 2) {  // Line format wrong
            cout << "Error: input behaviors file format wrong!" << endl;
            exit(1);
        } else {
            // If encounter new Goal
            if (_goals_set.find(tokens[0]) == _goals_set.end()) {
                goals_.push_back(tokens[0]);
                goal2goal_x.insert(make_pair(tokens[0], (lint)_goals_set.size()));
                _goals_set.insert(tokens[0]);
            }
            vector<lint> _context_xs_;
            stringstream ss_l2(tokens[1]);  // tokens[0] for Goal; tokens[1] for Plan (Contexts)
            string token_l2;
            while (getline(ss_l2, token_l2, delim_l2)) {
                // If encounter new Context
                if (_contexts_set.find(token_l2) == _contexts_set.end()) {
                    contexts_.push_back(token_l2);
                    context2context_x.insert(make_pair(token_l2, (lint)_contexts_set.size()));
                    _contexts_set.insert(token_l2);
                }
                _context_xs_.push_back(context2context_x[token_l2]);
            }
            sort(_context_xs_.begin(), _context_xs_.end());  // CRITICAL
            // If encounter new Plan
            if (_plans_set.find(tokens[1]) == _plans_set.end()) {
                plans_.push_back(_context_xs_);
                context_xs_sorted2plan_x.insert(make_pair(_context_xs_, (lint)_plans_set.size()));
                _plans_set.insert(tokens[1]);
            }
            // Add Goal, Plan pair
            vector<lint> _goal_x_plan_x_pr;
            _goal_x_plan_x_pr.push_back(goal2goal_x[tokens[0]]);
            _goal_x_plan_x_pr.push_back(context_xs_sorted2plan_x[_context_xs_]);
            goal_x_plan_x_prs_.push_back(_goal_x_plan_x_pr);
        }
    }
    num_goals = (lint)goals_.size();
    num_contexts = (lint)contexts_.size();
    num_plans = (lint)plans_.size();
    num_behaviors = (lint)goal_x_plan_x_prs_.size();

    // Print reading operation summary
    cout << "  #Goals: " << to_string(num_goals) << endl;
    cout << "  #Contexts: " << to_string(num_contexts) << endl;
    cout << "  #Plans: " << to_string(num_plans) << endl;
    cout << "  #Behaviors: " << to_string(num_behaviors) << endl;
}


/*
 * Initializations
 */
void initialize() {
    cout << "Initializing ..." << endl;

    /* Compute total_samples_num */
    if (total_samples_input != 0.0) {
        total_samples_num = (lint)(total_samples_input * total_samples_unit);
    } else {
        total_samples_num = goal_x_plan_x_prs_.size() * 500;
    }

    /* Randomly generate embeddings */
    random_device rd;
    // Random engines
    mt19937 engine(rd());
    // knuth_b engine(rd());
    // default_random_engine engine(rd()) ;

    real default_high_bound = 0.1;
    auto default_low_bound = -default_high_bound;
    uniform_real_distribution<real> dist(default_low_bound, default_high_bound);

    vector<real> _emb(static_cast<unsigned long>(emb_dim_num));
    // Goal embeddings
    for (lint i = 0; i < num_goals; i ++) {
        for (lint d = 0; d < emb_dim_num; d ++) {
            _emb[d] = dist(engine);
        }
        goal_x2emb_vec.push_back(_emb);
    }
    optimal_goal_x2emb_vec = goal_x2emb_vec;
    // Context embeddings
    for (lint i = 0; i < num_contexts; i ++) {
        for (lint d = 0; d < emb_dim_num; d ++) {
            _emb[d] = dist(engine);
        }
        context_x2emb_vec.push_back(_emb);
    }
    optimal_context_x2emb_vec = context_x2emb_vec;

    cout << "Done!" << endl;
}


/*
 * Write out optimal Goal, Context embeddings to external file
 */
void output_embs() {
    /* Output Goal embeddings */
    cout << "Writing Goal embeddings to " << goal_embs_file << " ..." << endl;
    ofstream goal_embs_fileout(goal_embs_file);
    // Write header line
    goal_embs_fileout << num_goals << "\t" << emb_dim_num << endl;
    // Write embeddings
    for (lint i = 0; i < num_goals; i ++) {
        goal_embs_fileout << goals_[i] << "\t";
        for (lint d = 0; d < emb_dim_num - 1 ; d ++) {
            goal_embs_fileout << optimal_goal_x2emb_vec[i][d] << "\t";
        }
        goal_embs_fileout << optimal_goal_x2emb_vec[i].back() << endl;
    }

    /* Output Context embeddings */
    cout << "Writing Context embeddings to " << context_embs_file << " ..." << endl;
    ofstream context_embs_fileout(context_embs_file);
    // Write header line
    context_embs_fileout << num_contexts << "\t" << emb_dim_num << endl;
    // Write embeddings
    for (lint i = 0; i < num_contexts; i ++) {
        context_embs_fileout << contexts_[i] << "\t";
        for (lint d = 0; d < emb_dim_num - 1 ; d ++) {
            context_embs_fileout << optimal_context_x2emb_vec[i][d] << "\t";
        }
        context_embs_fileout << optimal_context_x2emb_vec[i].back() << endl;
    }

    cout << "Done!" << endl;
}


/*
 * Parse arguments from command line
 */
string parse_cmd_args(int argc, char* argv[], const string& option) {
    string value;
    for (int i = 1; i < argc - 1; i++) {
        string arg = argv[i];
        size_t found_opt = arg.find(option);

        if (found_opt != string::npos) {  // If option is found
            string arg_next = argv[i + 1];

            if (arg_next.find("--") == string::npos) {  // Make sure value is not missing
                value = arg_next;
            }
        }
    }
    return value;
}


int main(int argc, char* argv[]) {
    /* Parse arguments */
    cout << "=============================================================================" << endl;
    // Required
    string arg_behaviors_file = parse_cmd_args(argc, argv, "--input_behaviors_file");
    string arg_goal_embs_file = parse_cmd_args(argc, argv, "--output_goal_embs_file");
    string arg_context_embs_file = parse_cmd_args(argc, argv, "--output_context_embs_file");
    if (arg_behaviors_file.empty() || arg_goal_embs_file.empty() || arg_context_embs_file.empty()) {
        cout << "Error: Required arguments cannot be empty!" << endl;
        exit(1);
    } else {
        behaviors_file = arg_behaviors_file;
        goal_embs_file = arg_goal_embs_file;
        context_embs_file = arg_context_embs_file;
    }
    // Optional
    string arg_emb_dim_num = parse_cmd_args(argc, argv, "--dims");
    string arg_threads_num = parse_cmd_args(argc, argv, "--threads");
    string arg_total_samples_input = parse_cmd_args(argc, argv, "--samples");
    string arg_negative_samples_num = parse_cmd_args(argc, argv, "--negative");
    string arg_init_learning_rate = parse_cmd_args(argc, argv, "--rate");

    if (! arg_emb_dim_num.empty()) {emb_dim_num = stol(arg_emb_dim_num);}
    if (! arg_threads_num.empty()) {threads_num = stol(arg_threads_num);}
    if (! arg_total_samples_input.empty()) {total_samples_input = stod(arg_total_samples_input);}
    if (! arg_negative_samples_num.empty()) {negative_samples_num = stol(arg_negative_samples_num);}
    if (! arg_init_learning_rate.empty()) {init_learning_rate = stod(arg_init_learning_rate);}

    // Print arguments
    cout << "Arguments:" << endl
         << "  --input_behaviors_file: " << behaviors_file << endl
         << "  --output_goal_embs_file: " << goal_embs_file << endl
         << "  --output_context_embs_file: " << context_embs_file << endl
         << "  --dims: " << emb_dim_num << endl
         << "  --threads: " << threads_num << endl
         << "  --samples (Millions): " << total_samples_input << endl
         << "  --negative: " << negative_samples_num << endl
         << "  --rate: " << init_learning_rate << endl;

    cout << "=============================================================================" << endl;

    using clock = chrono::steady_clock;
    /* Read in behaviors file & Initializations */
    auto t_i_s = clock::now();
    read_behaviors_file(behaviors_file, '\t', ',');
    initialize();
    auto t_i_e = clock::now();
    cout << "=============================================================================" << endl;

    /* Training */
    auto t_t_s = clock::now();
    train();
    auto t_t_e = clock::now();
    cout << "=============================================================================" << endl;

    /* Output embeddings */
    auto t_o_s = clock::now();
    output_embs();
    auto t_o_e = clock::now();
    cout << "=============================================================================" << endl;

    /* Print runtime summary */
    auto init_time = (real)chrono::duration_cast<chrono::milliseconds>(t_i_e-t_i_s).count();
    auto training_time = (real)chrono::duration_cast<chrono::milliseconds>(t_t_e-t_t_s).count();
    auto output_time = (real)chrono::duration_cast<chrono::milliseconds>(t_o_e-t_o_s).count();
    auto total_time = (real)chrono::duration_cast<chrono::milliseconds>(t_o_e-t_i_s).count();
    cout << fixed << setw(5) << setprecision(2) << "Total elapsed time: " << total_time/1000 << " s" << endl
         << "  Initialization: " << init_time/1000 << " s"
         << " (" << (init_time/total_time)*100 << "%)" << endl
         << "  Training: " << training_time/1000 << " s"
         << " (" << (training_time/total_time)*100 << "%)" << endl
         << "  Output: " << output_time/1000 << " s"
         << " (" << (output_time/total_time)*100 << "%)";
    cout << endl;
    cout << "=============================================================================" << endl;

    return 0;
}
