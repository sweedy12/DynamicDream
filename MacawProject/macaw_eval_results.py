import pandas as pd
import string

codah_gpt3_macaw_large_csv_path = 'gpt3_eval_results/dataset_codah_100_samples_test_gpt3_dim_elab_model_macaw-large_sample_size_-1_with_context_True_results_details.csv'
codah_no_context_macaw_large_csv_path = 'no_contexts_eval_results/dataset_codah_100_samples_test_gpt3_dim_elab_model_macaw-large_sample_size_-1_with_context_False_results_details.csv'
codah_gpt3_macaw_3b_csv_path = 'gpt3_eval_results/dataset_codah_100_samples_test_gpt3_dim_elab_model_macaw-3b_sample_size_-1_with_context_True_results_details.csv'
codah_no_context_macaw_3b_csv_path = 'no_contexts_eval_results/dataset_codah_100_samples_test_gpt3_dim_elab_model_macaw-3b_sample_size_-1_with_context_False_results_details.csv'
codah_no_context_macaw_3b_all_samples_csv_path = 'no_contexts_eval_results/dataset_codah_-1_samples_test_model_macaw-3b_sample_size_-1_with_context_False_results_details.csv'
codah_no_context_macaw_large_all_samples_csv_path = 'no_contexts_eval_results/dataset_codah_-1_samples_test_model_macaw-large_sample_size_-1_with_context_False_results_details.csv'

social_iqa_gpt3_macaw_large_csv_path = 'gpt3_eval_results/dataset_social_iqa_100_samples_test_gpt3_dim_elab_model_macaw-large_sample_size_-1_with_context_True_results_details.csv'
social_iqa_no_context_macaw_large_csv_path = 'no_contexts_eval_results/dataset_social_iqa_100_samples_test_gpt3_dim_elab_model_macaw-large_sample_size_-1_with_context_False_results_details.csv'
social_iqa_gpt3_macaw_3b_csv_path = 'gpt3_eval_results/dataset_social_iqa_100_samples_test_gpt3_dim_elab_model_macaw-3b_sample_size_-1_with_context_True_results_details.csv'
social_iqa_no_context_macaw_3b_csv_path = 'no_contexts_eval_results/dataset_social_iqa_100_samples_test_gpt3_dim_elab_model_macaw-3b_sample_size_-1_with_context_False_results_details.csv'
social_iqa_no_context_macaw_3b_all_samples_csv_path = 'no_contexts_eval_results/dataset_social_iqa_-1_samples_test_model_macaw-3b_sample_size_-1_with_context_False_results_details.csv'
social_iqa_no_context_macaw_large_all_samples_csv_path = 'no_contexts_eval_results/dataset_social_iqa_-1_samples_test_model_macaw-large_sample_size_-1_with_context_False_results_details.csv'

commonsense_qa_gpt3_macaw_large_csv_path = 'gpt3_eval_results/dataset_commonsense_qa_100_samples_test_gpt3_dim_elab_model_macaw-large_sample_size_-1_with_context_True_results_details.csv'
commonsense_qa_no_context_macaw_large_csv_path = 'no_contexts_eval_results/dataset_commonsense_qa_100_samples_test_gpt3_dim_elab_model_macaw-large_sample_size_-1_with_context_False_results_details.csv'
commonsense_qa_gpt3_macaw_3b_csv_path = 'gpt3_eval_results/dataset_commonsense_qa_100_samples_test_gpt3_dim_elab_model_macaw-3b_sample_size_-1_with_context_True_results_details.csv'
commonsense_qa_no_context_macaw_3b_csv_path = 'no_contexts_eval_results/dataset_commonsense_qa_100_samples_test_gpt3_dim_elab_model_macaw-3b_sample_size_-1_with_context_False_results_details.csv'
commonsense_qa_no_context_macaw_3b_all_samples_csv_path = 'no_contexts_eval_results/dataset_commonsense_qa_-1_samples_test_model_macaw-3b_sample_size_-1_with_context_False_results_details.csv'
commonsense_qa_no_context_macaw_large_all_samples_csv_path = 'no_contexts_eval_results/dataset_commonsense_qa_-1_samples_test_model_macaw-large_sample_size_-1_with_context_False_results_details.csv'


ethics_gpt3_macaw_large_csv_path = 'gpt3_eval_results/dataset_ethics_100_samples_test_gpt3_dim_elab_model_macaw-large_sample_size_-1_with_context_True_results_details.csv'
ethics_no_context_macaw_large_csv_path = 'no_contexts_eval_results/dataset_ethics_100_samples_test_gpt3_dim_elab_model_macaw-large_sample_size_-1_with_context_False_results_details.csv'
ethics_gpt3_macaw_3b_csv_path = 'gpt3_eval_results/dataset_ethics_100_samples_test_gpt3_dim_elab_model_macaw-3b_sample_size_-1_with_context_True_results_details.csv'
ethics_no_context_macaw_3b_csv_path = 'no_contexts_eval_results/dataset_ethics_100_samples_test_gpt3_dim_elab_model_macaw-3b_sample_size_-1_with_context_False_results_details.csv'
ethics_no_context_macaw_3b_all_samples_csv_path = 'no_contexts_eval_results/dataset_ethics_-1_samples_test_model_macaw-3b_sample_size_-1_with_context_False_results_details.csv'
ethics_no_context_macaw_large_all_samples_csv_path = 'no_contexts_eval_results/dataset_ethics_-1_samples_test_model_macaw-large_sample_size_-1_with_context_False_results_details.csv'


piqa_gpt3_macaw_large_csv_path = 'gpt3_eval_results/dataset_piqa_100_samples_test_gpt3_dim_elab_model_macaw-large_sample_size_-1_with_context_True_results_details.csv'
piqa_no_context_macaw_large_csv_path = 'no_contexts_eval_results/dataset_piqa_100_samples_test_gpt3_dim_elab_model_macaw-large_sample_size_-1_with_context_False_results_details.csv'
piqa_gpt3_macaw_3b_csv_path = 'gpt3_eval_results/dataset_piqa_100_samples_test_gpt3_dim_elab_model_macaw-3b_sample_size_-1_with_context_True_results_details.csv'
piqa_no_context_macaw_3b_csv_path = 'no_contexts_eval_results/dataset_piqa_100_samples_test_gpt3_dim_elab_model_macaw-3b_sample_size_-1_with_context_False_results_details.csv'
piqa_no_context_macaw_3b_all_samples_csv_path = 'no_contexts_eval_results/dataset_piqa_-1_samples_test_model_macaw-3b_sample_size_-1_with_context_False_results_details.csv'
piqa_no_context_macaw_large_all_samples_csv_path = 'no_contexts_eval_results/dataset_piqa_-1_samples_test_model_macaw-large_sample_size_-1_with_context_False_results_details.csv'


def print_eval_result(csv_filename):
    print(csv_filename)
    df = pd.read_csv(csv_filename)
    print(str(df["is_correct"].sum()) + " \ " + str(len(df)) + " which is: " + str(df["is_correct"].sum() / len(df) * 100) + " success rate")



if __name__ == "__main__":
    # Test the function
    text = "shoots the transgressor."
    print()



    df = pd.read_csv(piqa_gpt3_macaw_large_csv_path)
    correct_rows = df[df['is_correct'] == 1]
    wrong_rows = df[df['is_correct'] == 0]
    print(len(correct_rows))
    print(len(wrong_rows))

    print("100 samples:")
    print_eval_result(codah_gpt3_macaw_large_csv_path)
    print_eval_result(codah_no_context_macaw_large_csv_path)
    print_eval_result(codah_gpt3_macaw_3b_csv_path)
    print_eval_result(codah_no_context_macaw_3b_csv_path)
    print("all samples:")
    print_eval_result(codah_no_context_macaw_large_all_samples_csv_path)
    print_eval_result(codah_no_context_macaw_3b_all_samples_csv_path)

    print(" ------ ")
    print("100 samples:")
    print_eval_result(social_iqa_gpt3_macaw_large_csv_path)
    print_eval_result(social_iqa_no_context_macaw_large_csv_path)
    print_eval_result(social_iqa_gpt3_macaw_3b_csv_path)
    print_eval_result(social_iqa_no_context_macaw_3b_csv_path)
    print("all samples:")
    print_eval_result(social_iqa_no_context_macaw_large_all_samples_csv_path)
    print_eval_result(social_iqa_no_context_macaw_3b_all_samples_csv_path)

    print(" ------ ")
    print("100 samples:")
    print_eval_result(commonsense_qa_gpt3_macaw_large_csv_path)
    print_eval_result(commonsense_qa_no_context_macaw_large_csv_path)
    print_eval_result(commonsense_qa_gpt3_macaw_3b_csv_path)
    print_eval_result(commonsense_qa_no_context_macaw_3b_csv_path)
    print("all samples:")
    print_eval_result(commonsense_qa_no_context_macaw_large_all_samples_csv_path)
    print_eval_result(commonsense_qa_no_context_macaw_3b_all_samples_csv_path)

    print(" ------ ")
    print("100 samples:")
    print_eval_result(ethics_gpt3_macaw_large_csv_path)
    print_eval_result(ethics_no_context_macaw_large_csv_path)
    print_eval_result(ethics_gpt3_macaw_3b_csv_path)
    print_eval_result(ethics_no_context_macaw_3b_csv_path)
    print("all samples:")
    print_eval_result(ethics_no_context_macaw_large_all_samples_csv_path)
    print_eval_result(ethics_no_context_macaw_3b_all_samples_csv_path)

    print(" ------ ")
    print("100 samples:")
    print_eval_result(piqa_gpt3_macaw_large_csv_path)
    print_eval_result(piqa_no_context_macaw_large_csv_path)
    print_eval_result(piqa_gpt3_macaw_3b_csv_path)
    print_eval_result(piqa_no_context_macaw_3b_csv_path)
    print("all samples:")
    print_eval_result(piqa_no_context_macaw_large_all_samples_csv_path)
    print_eval_result(piqa_no_context_macaw_3b_all_samples_csv_path)








