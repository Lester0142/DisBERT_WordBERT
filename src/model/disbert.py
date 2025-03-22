import numpy as np
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel


class AutoModelForSequenceClassification_WSD_MIP_SPV(nn.Module):
    """ DisBERT Architecture """

    def __init__(self, args, Model, config, num_labels=2):
        """Initialize the model"""
        super(AutoModelForSequenceClassification_WSD_MIP_SPV, self).__init__()
        self.num_labels = num_labels
        self.encoder = Model
        self.config = config
        self.dropout = nn.Dropout(args.dropout_rate)
        self.args = args

        self.SPV_linear = nn.Linear(self.config.hidden_size * 2, self.args.classifier_hidden)
        self.MIP_linear = nn.Linear(self.config.hidden_size * 2, self.args.classifier_hidden)
        self.WSD_linear = nn.Linear(self.config.hidden_size * 2, self.args.classifier_hidden)
        self.classifier = nn.Linear(self.args.classifier_hidden * 3, self.num_labels)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self._init_weights(self.SPV_linear)
        self._init_weights(self.MIP_linear)
        self._init_weights(self.WSD_linear)
        self._init_weights(self.classifier)

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
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        head_mask=None,
    ):
        """
        Perform a forward pass through the model.

        Args:
            input_ids (torch.LongTensor): A tensor of shape [batch_size, sequence_length] containing token indices for the first input sequence.
            input_ids_2 (torch.LongTensor): A tensor of shape [batch_size, sequence_length] containing token indices for the second input sequence.
            target_mask (torch.LongTensor): A tensor of shape [batch_size, sequence_length] indicating the positions of the target word in the first input. (1 for the target word, 0 otherwise.)
            target_mask_2 (torch.LongTensor): A tensor of shape [batch_size, sequence_length] indicating the positions of the target word in the second input. (1 for the target word, 0 otherwise.)
            attention_mask_2 (torch.LongTensor): A tensor of shape [batch_size, sequence_length] with binary values [0, 1] indicating attention mask for the second input.
            token_type_ids (torch.LongTensor): A tensor of shape [batch_size, sequence_length] indicating token type indices (e.g., 0 for sentence A and 1 for sentence B).
            attention_mask (torch.LongTensor): A tensor of shape [batch_size, sequence_length] with binary values [0, 1] indicating attention mask for the first input.
            labels (torch.LongTensor): A tensor of shape [batch_size] containing the target labels for classification. If provided, the model will compute the loss.
            head_mask (torch.Tensor): A tensor indicating which attention heads to mask. Shape [num_heads] or [num_layers, num_heads].

        Returns:
            torch.Tensor: If labels are not provided, returns the predicted logits of shape [batch_size, num_labels]. 
                        If labels are provided, returns the loss value.

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

        # Second encoder for only the target word
        outputs_2 = self.encoder(input_ids_2, attention_mask=attention_mask_2, head_mask=head_mask)
        sequence_output_2 = outputs_2[0]  # [batch, max_len, hidden]

        # Get target ouput with target mask
        target_output_2 = sequence_output_2 * target_mask_2.unsqueeze(2)
        target_output_2 = self.dropout(target_output_2)
        target_output_2 = target_output_2.mean(1) # [batch, hidden]

        # Get hidden vectors each from SPV and MIP linear layers
        SPV_hidden = self.SPV_linear(torch.cat([pooled_output, target_output], dim=1))
        MIP_hidden = self.MIP_linear(torch.cat([target_output_2, target_output], dim=1))
        WSD_hidden = self.WSD_linear(torch.cat([pooled_output, target_output_2], dim=1))

        # Formulate Classifier input
        logits = self.classifier(self.dropout(torch.cat([SPV_hidden, MIP_hidden, WSD_hidden], dim=1)))

        # Softmax Classifier
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        
        return logits
