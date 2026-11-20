# from https://github.com/dmbeaglehole/neural_controllers, cite Beaglehole et al.
import torch, random, time, tqdm
from .direction_utils import *
from xrfm import RFM

my_verbose = False

def split_indices(N, frac=0.2, max_val_count=1024, random_split=True):
    n_train = N - min(int(frac*N), max_val_count)
    n_train = n_train + n_train%2 # ensure even train samples
    
    if random_split:
        indices = list(range(N))
        random.Random(15).shuffle(indices)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
    else:
        train_indices = range(n_train)
        val_indices = range(n_train, N)
    return train_indices, val_indices
        
def split_train_states(inputs, train_indices, val_indices):
    train_inputs, val_inputs = {}, {}
    for layer_idx, layer_states in inputs.items():
        train_inputs[layer_idx] = layer_states[train_indices]
        val_inputs[layer_idx] = layer_states[val_indices]
    return train_inputs, val_inputs

class Toolkit:
    def __init__(self):
        pass
        
    def preprocess_data(self, train_data, train_labels, val_data, val_labels, test_data, test_labels, 
                         model, tokenizer, hidden_layers, hyperparams, device='cuda'):
        """
        Handles preprocessing of train/val/test data and extracts hidden states.
        
        Returns:
            train_hidden_states: Dictionary mapping layer names to hidden states
            val_hidden_states: Dictionary mapping layer names to hidden states
            test_hidden_states: Dictionary mapping layer names to hidden states
            train_y: Training labels
            val_y: Validation labels
            test_y: Test labels
            test_data_provided: Boolean indicating if test data was provided
            num_classes: Number of classes in the labels
        """
        train_passed_as_hidden_states = isinstance(train_data, dict)
        val_passed_as_hidden_states = isinstance(val_data, dict)
        test_passed_as_hidden_states = isinstance(test_data, dict)

        if not isinstance(train_labels, torch.Tensor):
            train_labels = torch.tensor(train_labels).reshape(-1,1)
        if val_data is not None and not isinstance(val_labels, torch.Tensor):
            val_labels = torch.tensor(val_labels).reshape(-1,1)

        if len(train_labels.shape) == 1:
            train_labels = train_labels.unsqueeze(-1)
        if val_labels is not None and len(val_labels.shape) == 1:
            val_labels = val_labels.unsqueeze(-1)

        if val_data is None:
            # val data not provided, split train data into train and val
            if train_passed_as_hidden_states:
                assert -1 in train_data.keys(), "train_data must have a key -1 for the last layer"
                train_indices, val_indices = split_indices(len(train_data[-1]))
                train_data, val_data = split_train_states(train_data, train_indices, val_indices)
                val_passed_as_hidden_states = True
            else:
                train_indices, val_indices = split_indices(len(train_data))
                val_data = [train_data[i] for i in val_indices]
                train_data = [train_data[i] for i in train_indices]
                
            all_y = train_labels.float().to(device)
            train_y = all_y[train_indices]
            val_y = all_y[val_indices]
        else:
            train_y = train_labels.float().to(device)
            val_y = val_labels.float().to(device)

        test_data_provided = test_data is not None 
        num_classes = train_y.shape[1]
        
        # Extract hidden states
        if train_passed_as_hidden_states:
            print("Assuming train_data is already a dictionary of hidden states")
            train_hidden_states = train_data
        else:
            train_hidden_states = get_hidden_states(train_data, model, tokenizer, hidden_layers, hyperparams['forward_batch_size'])

        if val_passed_as_hidden_states:
            print("Assuming val_data is already a dictionary of hidden states")
            val_hidden_states = val_data
        else:
            val_hidden_states = get_hidden_states(val_data, model, tokenizer, hidden_layers, hyperparams['forward_batch_size'])

        test_hidden_states = None
        test_y = None
        if test_data_provided:
            if test_passed_as_hidden_states:
                print("Assuming test_data is already a dictionary of hidden states")
                test_hidden_states = test_data
            else:
                test_hidden_states = get_hidden_states(test_data, model, tokenizer, hidden_layers, hyperparams['forward_batch_size'])
            test_y = torch.tensor(test_labels).reshape(-1, num_classes).float().to(device)
        
        return (train_hidden_states, val_hidden_states, test_hidden_states, 
                train_y, val_y, test_y, test_data_provided, num_classes)
    
    def get_layer_data(self, layer_to_eval, train_hidden_states, val_hidden_states, train_y, val_y, device='cuda'):
        """
        Extracts data for a specific layer.
        
        Returns:
            train_X: Training data for the layer
            val_X: Validation data for the layer
        """

        train_X = train_hidden_states[layer_to_eval].float().to(device)
        val_X = val_hidden_states[layer_to_eval].float().to(device)
        if my_verbose:  
            print("train X shape:", train_X.shape, "train y shape:", train_y.shape, 
                "val X shape:", val_X.shape, "val y shape:", val_y.shape)
        assert(len(train_X) == len(train_y))
        assert(len(val_X) == len(val_y))
        
        return train_X, val_X
    
    def _compute_directions(self, train_data, train_labels, val_data, val_labels, model, tokenizer, hidden_layers, hyperparams,
                            test_data=None, test_labels=None, device='cuda', **kwargs):
        """
        Base implementation to be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _compute_directions")

class RFMToolkit(Toolkit):
    def __init__(self):
        super().__init__()

    def _compute_directions(self, train_data, train_labels, val_data, val_labels, model, tokenizer, hidden_layers, hyperparams,
                            test_data=None, test_labels=None, device='cuda', **kwargs):
        
        compare_to_linear = kwargs.get('compare_to_linear', False)
        log_spectrum = kwargs.get('log_spectrum', False)
        log_path = kwargs.get('log_path', None)
        tuning_metric = kwargs.get('tuning_metric', 'auc')

        print("Tuning metric:", tuning_metric)
        
        # Process data and extract hidden states
        (train_hidden_states, val_hidden_states, test_hidden_states, 
         train_y, val_y, test_y, test_data_provided, num_classes) = self.preprocess_data(
            train_data, train_labels, val_data, val_labels, test_data, test_labels, 
            model, tokenizer, hidden_layers, hyperparams, device
        )


        direction_outputs = {
            'train': [],
            'val': [],
            'test': []
        }
        
        if test_data_provided:
            test_direction_accs = {}
            test_predictor_accs = {}
        
        n_components = hyperparams['n_components']
        directions = {}
        detector_coefs = {}

        for layer_to_eval in tqdm(hidden_layers):
            # Get data for this layer
            train_X, val_X = self.get_layer_data(layer_to_eval, train_hidden_states, val_hidden_states, train_y, val_y, device)

            start_time = time.time()
            rfm_probe = train_rfm_probe_on_concept(train_X, train_y, val_X, val_y, hyperparams, tuning_metric=tuning_metric)
            end_time = time.time()
            if my_verbose:
                print(f"Time taken to train rfm probe: {end_time - start_time} seconds")
            if isinstance(rfm_probe, RFM):
                concept_features = rfm_probe.agop_best_model
            else:
                concept_features = rfm_probe.collect_best_agops()[0]

            if compare_to_linear:
                _ = train_linear_probe_on_concept(train_X, train_y, val_X, val_y)
    
            # S, U = torch.linalg.eigh(concept_features)
            start_time = time.time()
            S, U = torch.lobpcg(concept_features, k=n_components)
            end_time = time.time()
            if my_verbose:
                print(f"Time taken to compute eigenvectors: {end_time - start_time} seconds")

            if log_spectrum:
                spectrum_filename = log_path + f'_layer_{layer_to_eval}.pt'
                print("spectrum_filename", spectrum_filename)
                torch.save(S.cpu(), spectrum_filename)

            components = U.T
            directions[layer_to_eval] = components
            
            
            ### Generate direction accuracy
            # solve for slope, intercept on training data
            vec = directions[layer_to_eval].T
            projected_train = train_X@vec
            beta, b = linear_solve(projected_train, train_y)
                
            detector_coefs[layer_to_eval] = [beta, b]
            
            if test_data_provided:
                
                test_X = test_hidden_states[layer_to_eval].to(device).float()
                
                ### Generate predictor outputs
                test_preds = rfm_probe.predict(test_X)
                
                ### Generate predictor accuracy
                pred_acc = accuracy_fn(test_preds, test_y)
                test_predictor_accs[layer_to_eval] = pred_acc
                 
                
                ### Generate direction outputs    
                projected_train = train_X@vec         
                projected_val = val_X@vec
                projected_test = test_X@vec

                direction_outputs['train'].append(projected_train.reshape(-1,n_components))
                direction_outputs['val'].append(projected_val.reshape(-1,n_components))
                direction_outputs['test'].append(projected_test.reshape(-1,n_components))
                
                # evaluate slope, intercept on test data
                projected_preds = projected_test@beta + b
                projected_preds = projected_preds.reshape(-1,num_classes)
                
                assert(projected_preds.shape==test_y.shape)
                
                dir_acc = accuracy_fn(projected_preds, test_y)
                test_direction_accs[layer_to_eval] = dir_acc
                
                
        signs = {}
        if num_classes == 1: # only if binary do you compute signs
            signs = self._compute_signs(train_hidden_states, train_y, directions, n_components)
            for layer_to_eval in tqdm(hidden_layers):
                for c_idx in range(n_components):
                    directions[layer_to_eval][c_idx] *= signs[layer_to_eval][c_idx]
                
        
        if test_data_provided:
            direction_agg_acc = aggregate_layers(direction_outputs, train_y, val_y, test_y)
            test_direction_accs['aggregated'] = direction_agg_acc
        
            return directions, signs, detector_coefs, test_direction_accs
        else: 
            return directions, signs, detector_coefs, None
        
    def _compute_signs(self, hidden_states, all_y, directions, n_components):
        signs = {}
        for layer in hidden_states.keys():
            xs = hidden_states[layer]
            signs[layer] = {}
            for c_idx in range(n_components):
                direction = directions[layer][c_idx]
                hidden_state_projections = project_onto_direction(xs, direction).to(all_y.device)
                sign = 2*(pearson_corr(all_y.squeeze(1), hidden_state_projections) > 0) - 1
                signs[layer][c_idx] = sign.item()

        return signs

