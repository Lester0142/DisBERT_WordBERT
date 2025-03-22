import numpy as np
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel


class AutoModelForSequenceClassification_DEF(nn.Module):
    """ MelBERT with a basic definition encoder """

    def __init__(self, args, Model, config, num_labels=2):
        """Initialize the model"""
        super(AutoModelForSequenceClassification_DEF, self).__init__()
        self.num_labels = num_labels
        self.encoder = Model
        self.config = config
        self.dropout = nn.Dropout(args.dropout_rate)
        self.args = args

        self.SPV_linear = nn.Linear(self.config.hidden_size * 2, self.args.classifier_hidden)
        self.MIP_linear = nn.Linear(self.config.hidden_size * 2, self.args.classifier_hidden)
        self.classifier = nn.Linear(self.args.classifier_hidden * 2, self.num_labels)

        self._init_weights(self.SPV_linear)
        self._init_weights(self.MIP_linear)
        self._init_weights(self.classifier)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Bilinear)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        input_ids_2,
        target_mask,
        target_mask_2,
        attention_mask_2,
        input_ids_3=None,
        attention_mask_3 = None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        head_mask=None,
    ):
        """
        Perform a forward pass through the model, processing multiple inputs to predict the target labels.

        Args:
            input_ids (torch.LongTensor): Tensor of shape [batch_size, sequence_length] containing token indices for the first input sequence.
            input_ids_2 (torch.LongTensor): Tensor of shape [batch_size, sequence_length] containing token indices for the second input sequence.
            target_mask (torch.LongTensor): Tensor of shape [batch_size, sequence_length] indicating the position of the target word in the first input (1 for target word, 0 otherwise).
            target_mask_2 (torch.LongTensor): Tensor of shape [batch_size, sequence_length] indicating the position of the target word in the second input (1 for target word, 0 otherwise).
            attention_mask_2 (torch.LongTensor): Tensor of shape [batch_size, sequence_length] with binary values [0, 1] indicating attention mask for the second input.
            input_ids_3 (torch.LongTensor): Tensor of shape [batch_size, sequence_length] for the third input sequence, containing basic word meaning or context.
            attention_mask_3 (torch.LongTensor): Tensor of shape [batch_size, sequence_length] with binary values [0, 1] for the attention mask of the third input.
            token_type_ids (torch.LongTensor): Tensor of shape [batch_size, sequence_length] with token type indices, where 0 indicates sentence A and 1 indicates sentence B.
            attention_mask (torch.LongTensor): Tensor of shape [batch_size, sequence_length] with binary values [0, 1] indicating attention mask for the first input.
            labels (torch.LongTensor): Tensor of shape [batch_size] containing the target labels for classification. Used to compute the loss.
            head_mask (torch.Tensor): Tensor for masking attention heads, with values between 0 and 1. Shape [num_heads] or [num_layers, num_heads].

        Returns:
            torch.Tensor: If labels are not provided, returns the predicted logits of shape [batch_size, num_labels]. 
                        If labels are provided, returns the loss value based on the predicted logits and the true labels.
        """

        # First encoder for full sentence
        outputs = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]  # [batch, max_len, hidden]
        pooled_output = outputs[1]  # [batch, hidden]

        # Get target ouput with target mask
        target_output = sequence_output * target_mask.unsqueeze(2)

        # dropout
        target_output = self.dropout(target_output)
        pooled_output = self.dropout(pooled_output)
        target_output = target_output.mean(1)  # [batch, hidden]

        # Third encoder for basic meaning of target word
        outputs_3 = self.encoder(input_ids_3, attention_mask=attention_mask_3)
        pooled_output_3 = outputs_3[1]
        pooled_output_3 = self.dropout(pooled_output_3) # [batch, hidden]

        # Get hidden vectors each from SPV and MIP linear layers
        SPV_hidden = self.SPV_linear(torch.cat([pooled_output, target_output], dim=1))
        MIP_hidden = self.MIP_linear(torch.cat([pooled_output_3, target_output], dim=1))

        # Formulate Classifier input
        logits = self.classifier(self.dropout(torch.cat([SPV_hidden, MIP_hidden], dim=1)))
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        
        return logits
    