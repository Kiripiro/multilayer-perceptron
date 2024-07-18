import argparse
from srcs.split import split_dataset
from srcs.train import run_training
from srcs.predict import evaluate_model
from srcs.histograms import histograms
import os
from colorama import init, Fore, Style
import random

init(autoreset=True)

def get_random_state(seed):
    if seed == 'rand':
        return random.randint(0, 10000)
    else:
        try:
            return int(seed)
        except ValueError:
            raise ValueError(f"Invalid random_state value: {seed}. It should be an integer or 'rand'.")

def main():
    try:
        parser = argparse.ArgumentParser(
            description=f"""
            {Fore.CYAN}Multilayer Perceptron for Breast Cancer Classification.{Style.RESET_ALL}
            This tool allows you to split the dataset, train a neural network model, and make predictions.

            {Fore.YELLOW}Available commands:{Style.RESET_ALL}
            - {Fore.GREEN}histograms{Style.RESET_ALL}: Visualize data repartition
            - {Fore.GREEN}split{Style.RESET_ALL}: Split the dataset into training and validation sets.
            - {Fore.GREEN}train{Style.RESET_ALL}: Train the neural network model.
            - {Fore.GREEN}predict{Style.RESET_ALL}: Make predictions using the trained model.
            - {Fore.GREEN}full{Style.RESET_ALL}: Run the entire process: split, train, and predict.
            """,
            formatter_class=argparse.RawTextHelpFormatter
        )
        subparsers = parser.add_subparsers(dest='command')

        parser_histogram = subparsers.add_parser(
            'histograms',
            help=f'{Fore.GREEN}Visualize data repartition.{Style.RESET_ALL} Example: python script.py histograms'
        )

        # Split sub-command
        parser_split = subparsers.add_parser(
            'split', 
            help=f'{Fore.GREEN}Split the dataset into training and validation sets.{Style.RESET_ALL} Example: python script.py split --input_csv data.csv --test_size 0.2 --random_state 42'
        )
        parser_split.add_argument(
            '--input_csv', 
            required=True, 
            help='Path to the input CSV file containing the dataset to be split.'
        )
        parser_split.add_argument(
            '--test_size', 
            type=float, 
            default=0.2, 
            help='Proportion of the dataset to include in the test split. Default is 0.2.'
        )
        parser_split.add_argument(
            '--random_state', 
            type=str, 
            default='32', 
            help='Seed used by the random number generator. Accepts int or "rand" for a random seed. Default is 32.'
        )

        # Train sub-command
        parser_train = subparsers.add_parser(
            'train', 
            help=f'{Fore.GREEN}Train the neural network model.{Style.RESET_ALL} Example: python script.py train --epochs 100 --optimizer adam --learning_rate 0.001 --early_stopping --patience 10'
        )
        parser_train.add_argument(
            '--epochs', 
            type=int, 
            default=100, 
            help='Number of training epochs. Default is 100.'
        )
        parser_train.add_argument(
            '--optimizer', 
            choices=['sgd', 'sgdMomentum', 'adam'], 
            default='sgd', 
            help='Optimizer to use for training. Choices are "sgd", "sgdMomentum", and "adam". Default is "sgd".'
        )
        parser_train.add_argument(
            '--learning_rate', 
            type=float, 
            default=0.01, 
            help='Learning rate for the optimizer. Default is 0.01.'
        )
        parser_train.add_argument(
            '--batch_size',
            default=32,
            type=int,
            help='Size of each mini-batch for training. Default is 32.'
        )
        parser_train.add_argument(
            '--early_stopping', 
            action='store_true', 
            help='Enable early stopping to prevent overfitting. If enabled, training will stop if the validation loss does not improve for a specified number of epochs (patience).'
        )
        parser_train.add_argument(
            '--patience', 
            type=int, 
            default=10, 
            help='Number of epochs with no improvement after which training will be stopped if early stopping is enabled. Default is 10.'
        )

        # Predict sub-command
        parser_predict = subparsers.add_parser(
            'predict', 
            help=f'{Fore.GREEN}Make predictions using the trained model.{Style.RESET_ALL} Example: python script.py predict --test_csv test_data.csv'
        )
        parser_predict.add_argument(
            '--test_csv', 
            default='data/train_test/test.csv',
            help='Path to the test CSV file containing the data for making predictions.'
        )
        parser_predict.add_argument(
            '--activations', 
            required=False,
            action='store_true',
            help='Plot the neuron\'s activations.'
        )

        # Full run sub-command
        parser_full = subparsers.add_parser(
            'full', 
            help=f'{Fore.GREEN}Run all the program at once{Style.RESET_ALL}: split the dataset, train the model, and make predictions. Example: python script.py full --input_csv data.csv --test_size 0.2 --random_state 42 --epochs 100 --optimizer adam --learning_rate 0.001 --early_stopping --patience 10 --test_csv test_data.csv'
        )
        parser_full.add_argument(
            '--input_csv', 
            required=True, 
            help='Path to the input CSV file containing the dataset to be split.'
        )
        parser_full.add_argument(
            '--test_csv', 
            default='data/train_test/test.csv',
            help='Path to the test CSV file containing the data for making predictions.'
        )
        parser_full.add_argument(
            '--test_size', 
            type=float,
            default=0.2,
            help='Proportion of the dataset to include in the test split. Default is 0.2.'
        )
        parser_full.add_argument(
            '--random_state', 
            type=str, 
            default='32', 
            help='Seed used by the random number generator. Accepts int or "rand" for a random seed. Default is 32.'
        )
        parser_full.add_argument(
            '--epochs', 
            type=int, 
            default=100, 
            help='Number of training epochs. Default is 100.'
        )
        parser_full.add_argument(
            '--optimizer', 
            choices=['sgd', 'sgdMomentum', 'adam'], 
            default='sgd', 
            help='Optimizer to use for training. Choices are "sgd", "sgdMomentum", and "adam". Default is "sgd".'
        )
        parser_full.add_argument(
            '--learning_rate', 
            type=float, 
            default=0.01, 
            help='Learning rate for the optimizer. Default is 0.01.'
        )
        parser_full.add_argument(
            '--early_stopping', 
            action='store_true', 
            help='Enable early stopping to prevent overfitting. If enabled, training will stop if the validation loss does not improve for a specified number of epochs (patience).'
        )
        parser_full.add_argument(
            '--patience', 
            type=int, 
            default=10, 
            help='Number of epochs with no improvement after which training will be stopped if early stopping is enabled. Default is 10.'
        )
        parser_full.add_argument(
            '--batch_size',
            default=32,
            type=int,
            help='Size of each mini-batch for training. Default is 32.'
        )
        parser_full.add_argument(
            '--activations', 
            required=False, 
            action='store_true',
            help='Plot the neuron\'s activations.'
        )

        args = parser.parse_args()

        if not vars(args).get('command'):
            parser.print_help()
            return

        if hasattr(args, 'input_csv') and not os.path.exists(args.input_csv):
            raise FileNotFoundError(f"{Fore.RED}Input CSV file not found: {args.input_csv}{Style.RESET_ALL}")
        if hasattr(args, 'test_csv') and not os.path.exists(args.test_csv):
            raise FileNotFoundError(f"{Fore.RED}Test CSV file not found: {args.test_csv}{Style.RESET_ALL}")

        if args.command in ['split', 'full']:
            random_state = get_random_state(args.random_state)

        if args.command == 'histograms':
            print(f"{Fore.BLUE}Exploring dataset...{Style.RESET_ALL}")
            histograms()
        elif args.command == 'split':
            print(f"{Fore.BLUE}Splitting the dataset...{Style.RESET_ALL}\n")
            split_dataset(args.input_csv, args.test_size, random_state)
            print(f"{Fore.GREEN}Dataset split completed.{Style.RESET_ALL}\n")
        elif args.command == 'train':
            print(f"{Fore.BLUE}Training the model...{Style.RESET_ALL}\n")
            run_training(args.epochs, args.optimizer, args.learning_rate, args.early_stopping, args.patience, args.batch_size)
            print(f"{Fore.GREEN}Model training completed.{Style.RESET_ALL}\n")
        elif args.command == 'predict':
            print(f"{Fore.BLUE}Making predictions...{Style.RESET_ALL}\n")
            evaluate_model(args.test_csv, args.activations)
            print(f"{Fore.GREEN}Predictions completed.{Style.RESET_ALL}\n")
        elif args.command == 'full':
            print(f"{Fore.BLUE}Running full pipeline: split, train, predict...{Style.RESET_ALL}\n")
            split_dataset(args.input_csv, args.test_size, random_state)
            print(f"{Fore.GREEN}Dataset split completed.{Style.RESET_ALL}\n")
            run_training(args.epochs, args.optimizer, args.learning_rate, args.early_stopping, args.patience, args.batch_size)
            print(f"{Fore.GREEN}Model training completed.{Style.RESET_ALL}\n")
            evaluate_model(args.test_csv, args.activations)
            print(f"{Fore.GREEN}Full pipeline completed.{Style.RESET_ALL}\n")
        else:
            parser.print_help()
    except FileNotFoundError as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
    except ValueError as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}An unexpected error occurred: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
