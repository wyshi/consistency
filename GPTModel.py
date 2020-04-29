from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer, GPT2Config#, Block
from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn
import pdb

class GPT2LMHeadModel_modified(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model_modified(config)
        # self.past_max_len = config.n_ctx
        # print(f"max_past_len = {self.past_max_len}")

    # def forward(
    #     self,
    #     input_ids=None,
    #     past=None,
    #     attention_mask=None,
    #     token_type_ids=None,
    #     position_ids=None,
    #     head_mask=None,
    #     inputs_embeds=None,
    #     labels=None,
    # ):

    #     input_ids_temp = input_ids
    #     inputs_embeds_temp = inputs_embeds
    #     token_type_ids_temp = token_type_ids

    #     if past is not None:
    #         if input_ids is not None:
    #             input_ids_temp = input_ids[:, -1:]
    #         if inputs_embeds is not None:
    #             inputs_embeds_temp = inputs_embeds[:, -1:]
    #         if token_type_ids is not None:
    #             token_type_ids_temp = token_type_ids[:, -1:]
            
    #     if input_ids_temp is not None and inputs_embeds_temp is not None:
    #         raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    #     elif input_ids_temp is not None:
    #         input_shape = input_ids_temp.size()
    #         input_ids_temp = input_ids_temp.view(-1, input_shape[-1])
    #         batch_size = input_ids_temp.shape[0]
    #     elif inputs_embeds_temp is not None:
    #         input_shape = inputs_embeds_temp.size()[:-1]
    #         batch_size = inputs_embeds_temp.shape[0]
    #     else:
    #         raise ValueError("You have to specify either input_ids or inputs_embeds")

    #     if past is None:
    #         past_length = 0
    #         # past = [None] * len(self.transformer.h)
    #     else:
    #         past_length = past[0][0].size(-2)

    #     import pdb
        

    #     if input_shape[-1] + past_length >= self.past_max_len:
    #         pdb.set_trace()
    #         truncated_past = [p for p in past]
 
    #     try:
    #         transformer_outputs = self.transformer(
    #             input_ids,
    #             past=past,
    #             attention_mask=attention_mask,
    #             token_type_ids=token_type_ids,
    #             position_ids=position_ids,
    #             head_mask=head_mask,
    #             inputs_embeds=inputs_embeds,
    #             # use_cache=use_cache,
    #         )
    #     except:
    #         pdb.set_trace()
    #     hidden_states = transformer_outputs[0]

    #     lm_logits = self.lm_head(hidden_states)

    #     outputs = (lm_logits,) + transformer_outputs[1:]
    #     if labels is not None:
    #         # Shift so that tokens < n predict n
    #         shift_logits = lm_logits[..., :-1, :].contiguous()
    #         shift_labels = labels[..., 1:].contiguous()
    #         # Flatten the tokens
    #         loss_fct = CrossEntropyLoss()
    #         loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    #         outputs = (loss,) + outputs

    #     return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)


class GPT2Model_modified(GPT2Model):
    def __init__(self, config, split_into=4):
        super().__init__(config)
        self.past_max_len = config.n_ctx
        print(f"max_past_len = {self.past_max_len}")

    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import GPT2Tokenizer, GPT2Model
        import torch

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            if input_shape[-1] + past_length > self.past_max_len:
                past = [p[:, :, :, -(self.past_max_len-input_shape[-1]):, :] for p in past]
                position_ids = torch.arange(self.past_max_len - input_shape[-1], self.past_max_len, dtype=torch.long, device=device)
            else:
                position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, input_shape[-1])
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(
                hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask[i]
            )

            hidden_states, present = outputs[:2]
            if self.output_past:
                presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_past:
            outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
        return outputs  # last hidden state, (presents), (all hidden_states), (attentions)
