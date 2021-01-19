Refrence https://pytorch.org/tutorials/beginner/fgsm_tutorial.html#threat-model
# Adversial_example_Generation
Fast Gradient Sign Attack
    the attack uses the gradient of the loss w.r.t the input data, then adjusts the input data to maximize the loss.
    
## Implementation    
    In this section, we will discuss the input parameters for the tutorial, define the model under attack, then code the attack and run some tests.    
Inputs
    There are only three inputs for this tutorial, and are defined as follows:

    epsilons - List of epsilon values to use for the run. It is important to keep 0 in the list because it represents the model performance on the original test set. Also, intuitively we would expect the larger the epsilon, the more noticeable the perturbations but the more effective the attack in terms of degrading model accuracy. Since the data range here is [0,1], no epsilon value should exceed 1.
    pretrained_model - path to the pretrained MNIST model which was trained with pytorch/examples/mnist. For simplicity, download the pretrained model here.
    use_cuda - boolean flag to use CUDA if desired and available. Note, a GPU with CUDA is not critical for this tutorial as a CPU will not take much time.
pretrained_model - path to the pretrained MNIST model which was trained with pytorch/examples/mnist. For simplicity, download the pretrained model [Here](https://drive.google.com/drive/folders/1fn83DF14tWmit0RTKWRhPq5uVXt73e0h?usp=sharing).
## Model Under Attack
    As mentioned, the model under attack is the same MNIST model from pytorch/examples/mnist. You may train and save your own MNIST model or you can download and use the provided model. The Net definition and test dataloader here have been copied from the MNIST example.
## FGSM Attack
    Now, we can define the function that creates the adversarial examples by perturbing the original inputs. The fgsm_attack function takes three inputs, image is the original clean image (x), epsilon is the pixel-wise perturbation amount (ϵ), and data_grad is gradient of the loss w.r.t the input image (∇xJ(θ,x,y)).
## Testing Function    
    the central result of this tutorial comes from the test function. Each call to this test function performs a full test step on the MNIST test set and reports a final accuracy.
    
    
## Run Attack

  The last part of the implementation is to actually run the attack. Here, we run a full test step for each epsilon value in the epsilons input. For each epsilon we also save the final accuracy and some successful adversarial examples to be plotted in the coming sections. Notice how the printed accuracies decrease as the epsilon value increases. Also, note the ϵ=0 case represents the original test accuracy, with no attack.

    accuracies = []
    examples = []

    # Run test for each epsilon
    for eps in epsilons:
        acc, ex = test(model, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)
        
## Results
### Accuracy vs Epsilon
    plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()
    
    
    
## Sample Adversarial Examples
    # Plot several examples of adversarial samples at each epsilon
    cnt = 0
    plt.figure(figsize=(8,10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons),len(examples[0]),cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig,adv,ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()
