#include <torch/torch.h>
#include <iostream>
#include <vector>

struct AutoencoderImpl : torch::nn::Module {
    // Layers
    torch::nn::Sequential encoder{nullptr}, decoder{nullptr};

    AutoencoderImpl(int input_dim, int latent_dim) {
        // Encoder
        encoder = torch::nn::Sequential(
            torch::nn::Linear(input_dim, 128),
            torch::nn::ReLU(),
            torch::nn::Linear(128, latent_dim)
        );
        register_module("encoder", encoder);

        // Decoder
        decoder = torch::nn::Sequential(
            torch::nn::Linear(latent_dim, 128),
            torch::nn::ReLU(),
            torch::nn::Linear(128, input_dim)
        );
        register_module("decoder", decoder);
    }

    torch::Tensor forward(torch::Tensor x) {
        auto latent = encoder->forward(x);
        auto reconstruction = decoder->forward(latent);
        return reconstruction;
    }
};

// Register the module
TORCH_MODULE(Autoencoder);

void train_autoencoder(Autoencoder& model, const torch::Tensor& data, int epochs, int batch_size, float learning_rate) {
    model->train();
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));
    auto criterion = torch::nn::MSELoss();

    auto dataset = torch::data::datasets::TensorDataset(data).map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader(dataset, batch_size);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        float epoch_loss = 0.0;
        for (auto& batch : *data_loader) {
            auto input = batch.data;

            // Forward pass
            auto reconstruction = model->forward(input);
            auto loss = criterion(reconstruction, input);

            // Backward pass and optimization
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            epoch_loss += loss.item<float>();
        }

        std::cout << "Epoch [" << epoch << "/" << epochs << "], Loss: " << epoch_loss / dataset.size().value() << "\n";
    }
}

