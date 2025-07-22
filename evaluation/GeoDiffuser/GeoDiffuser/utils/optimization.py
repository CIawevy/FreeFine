import torch
from GeoDiffuser.utils.generic_torch import *




def adaptive_optimization_step_editing(controller, i, skip_optim_steps, out_loss_log_dict, num_ddim_steps, removal_loss_value_in = -1.5):


    print(f"[INFO]: Adaptive Removal Loss value {removal_loss_value_in}")

    if (i / num_ddim_steps) < 0.4:
        remaining_steps = int((0.4 - (i / num_ddim_steps)) * num_ddim_steps / skip_optim_steps)
        expected_removal_loss_value = removal_loss_value_in / (1.25)**(remaining_steps)
        expected_movement_loss_value = 0.4 / (1.1)**(remaining_steps)

        expected_sim_loss_value = 0.5 / (1.05)**(remaining_steps)

        print(i, " remaining_steps: ", remaining_steps, " expected removal loss: ", expected_removal_loss_value, " current loss: ", out_loss_log_dict["self"]["removal"])
        # print(i, " remaining_steps: ", remaining_steps, " expected movement loss: ", expected_movement_loss_value, " current loss: ", out_loss_log_dict["self"]["movement"])
        # print(i, " remaining_steps: ", remaining_steps, " expected movement loss: ", expected_sim_loss_value, " current loss: ", out_loss_log_dict["self"]["sim"])

        # if (expected_movement_loss_value < out_loss_log_dict["self"]["movement"]):
        #     print("increasing movement loss weight")
        #     controller.loss_weight_dict["self"]["movement"] *= 1.5
        
        # if (expected_sim_loss_value < out_loss_log_dict["self"]["sim"]):
        #     print("increasing sim loss weight")
            # controller.loss_weight_dict["self"]["sim"] *= 1.05
        # elif (expected_movement_loss_value > out_loss_log_dict["self"]["movement"]):
        #     controller.loss_weight_dict["self"]["movement"] *= 0.5



        if (expected_removal_loss_value < out_loss_log_dict["self"]["removal"]):
            # print(controller.loss_weight_dict)
            print("increasing removal loss weight")
            controller.loss_weight_dict["self"]["removal"] *= 1.3
            # print(controller.loss_weight_dict)

        elif (2.5 * expected_removal_loss_value > out_loss_log_dict["self"]["removal"]):
            # print(controller.loss_weight_dict)
            print("reducing removal loss weight")
            controller.loss_weight_dict["self"]["removal"] /= 2.0
            # print(controller.loss_weight_dict)
        
    elif ((i / num_ddim_steps) > 0.4) and ((i / num_ddim_steps) < 0.8):

        if ((removal_loss_value_in - 0.3) < out_loss_log_dict["self"]["removal"]):
            controller.loss_weight_dict["self"]["removal"] *= 2.0
        else:
            controller.initialize_default_loss_weights()

    else:
        controller.initialize_default_loss_weights()
        # print("reinitialized controller loss weights: ", controller.loss_weight_dict)

def adaptive_optimization_step_remover(controller, i, skip_optim_steps, out_loss_log_dict, num_ddim_steps, removal_loss_value_in = -1.5):

    print(f"[INFO]: Adaptive Removal Loss value {removal_loss_value_in}")


    if (i / num_ddim_steps) < 0.4:
        remaining_steps = int((0.4 - (i / num_ddim_steps)) * num_ddim_steps / skip_optim_steps)
        expected_removal_loss_value = removal_loss_value_in / (1.25)**(remaining_steps)
        expected_movement_loss_value = 0.4 / (1.1)**(remaining_steps)

        expected_sim_loss_value = 0.5 / (1.05)**(remaining_steps)

        print(i, " remaining_steps: ", remaining_steps, " expected removal loss: ", expected_removal_loss_value, " current loss: ", out_loss_log_dict["self"]["removal"])
        
        # if (expected_movement_loss_value < out_loss_log_dict["self"]["movement"]):
        #     print("increasing movement loss weight")
        #     controller.loss_weight_dict["self"]["movement"] *= 1.5
        
        # if (expected_sim_loss_value < out_loss_log_dict["self"]["sim"]):
        #     print("increasing sim loss weight")
            # controller.loss_weight_dict["self"]["sim"] *= 1.05
        # elif (expected_movement_loss_value > out_loss_log_dict["self"]["movement"]):
        #     controller.loss_weight_dict["self"]["movement"] *= 0.5



        if (expected_removal_loss_value < out_loss_log_dict["self"]["removal"]):
            # print(controller.loss_weight_dict)
            print("increasing removal loss weight")
            controller.loss_weight_dict["self"]["removal"] *= 1.3
            # print(controller.loss_weight_dict)

        elif (2.5 * expected_removal_loss_value > out_loss_log_dict["self"]["removal"]):
            # print(controller.loss_weight_dict)
            print("reducing removal loss weight")
            controller.loss_weight_dict["self"]["removal"] /= 2.5
            # print(controller.loss_weight_dict)
        
    elif ((i / num_ddim_steps) > 0.4) and ((i / num_ddim_steps) < 0.8):

        if ((removal_loss_value_in - 0.3) < out_loss_log_dict["self"]["removal"]):
            controller.loss_weight_dict["self"]["removal"] *= 2.0
        else:
            controller.initialize_default_loss_weights()

    else:
        controller.initialize_default_loss_weights()
        # print("reinitialized controller loss weights: ", controller.loss_weight_dict)



def adaptive_optimization_step_stitching(controller, i, skip_optim_steps, out_loss_log_dict, num_ddim_steps):


    # if (i / num_ddim_steps) > 0.4 and ((i / num_ddim_steps) < 0.45):
    #     controller.loss_weight_dict["self"]["sim_out"] /= 1.5
    #     controller.loss_weight_dict["cross"]["sim_out"] /= 1.5
    #     print("Reducing sim loss")
    
    # return
    if (i / num_ddim_steps) < 0.4:
        remaining_steps = int((0.4 - (i / num_ddim_steps)) * num_ddim_steps / skip_optim_steps)
        expected_sim_loss_value = 0.18 / (1.01)**(remaining_steps)
        # expected_movement_loss_value = 0.4 / (1.1)**(remaining_steps)

        # expected_sim_loss_value = 0.5 / (1.05)**(remaining_steps)

        # print(i, " remaining_steps: ", remaining_steps, " expected removal loss: ", expected_removal_loss_value, " current loss: ", out_loss_log_dict["self"]["removal"])
        # print(i, " remaining_steps: ", remaining_steps, " expected movement loss: ", expected_movement_loss_value, " current loss: ", out_loss_log_dict["self"]["movement"])
        print(i, " remaining_steps: ", remaining_steps, " expected movement loss: ", expected_sim_loss_value, " current loss: ", out_loss_log_dict["self"]["sim_out"])

        # if (expected_movement_loss_value < out_loss_log_dict["self"]["movement"]):
        #     print("increasing movement loss weight")
        #     controller.loss_weight_dict["self"]["movement"] *= 1.5
        
        # if (expected_sim_loss_value < out_loss_log_dict["self"]["sim"]):
        #     print("increasing sim loss weight")
            # controller.loss_weight_dict["self"]["sim"] *= 1.05
        # elif (expected_movement_loss_value > out_loss_log_dict["self"]["movement"]):
        #     controller.loss_weight_dict["self"]["movement"] *= 0.5



        if (expected_sim_loss_value < out_loss_log_dict["self"]["sim_out"]):
            # print(controller.loss_weight_dict)
            print("increasing sim loss weight")
            controller.loss_weight_dict["self"]["sim_out"] *= 1.1
            # print(controller.loss_weight_dict)

        elif (2.5 * expected_sim_loss_value > out_loss_log_dict["self"]["sim_out"]):
            # print(controller.loss_weight_dict)
            print("reducing sim loss weight")
            controller.loss_weight_dict["self"]["sim_out"] /= 2.5
            # print(controller.loss_weight_dict)
        
    elif ((i / num_ddim_steps) > 0.4) and ((i / num_ddim_steps) < 0.7):

        if (0.2 < out_loss_log_dict["self"]["sim_out"]):
            controller.loss_weight_dict["self"]["sim_out"] *= 1.1
        else:
            controller.initialize_default_loss_weights()

    else:
        controller.initialize_default_loss_weights()
        # print("reinitialized controller loss weights: ", controller.loss_weight_dict)


def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float, mask = None, context = None, scaler = None, optimizer = None) -> torch.Tensor:
    """ Update the latent according to the computed loss. """
    
    
    if context is not None:

        # loss_new = scaler.scale(loss)
        # https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-unscaled-gradients
        loss_new  = loss
        if scaler is not None:
            loss_new = scaler.scale(loss)    
        



        if optimizer is None:
        # torch.autograd.backward(loss_new, inputs=[latents, context], retain_graph=False)
            grads = torch.autograd.grad(loss_new, [latents, context], retain_graph = False)
        
        else:
            print("[INFO]: Using PyTorch Optimizer")
            optimizer.zero_grad()
            loss_new.backward()
            if scaler is not None:
                print("[INFO]: Using Scaler")
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            

            latents_in, context_new = optimizer.param_groups[0]["params"]
            latents_out = latents_in.detach().clone()
            context_out = context_new.detach().clone()

            latents_out[torch.logical_not(torch.isfinite(latents_out))] = latents[torch.logical_not(torch.isfinite(latents_out))].detach()
            context_out[torch.logical_not(torch.isfinite(context_out))] = context[torch.logical_not(torch.isfinite(context_out))].detach() 
            # latents_in = ((1.0 * torch.isfinite(latents_in)) * latents_in + (1.0 - (1.0 * torch.isfinite(latents_in))) * latents.detach()).detach()
            # context_new = ((1.0 * torch.isfinite(context_new)) * context_new + (1.0 - (1.0 * torch.isfinite(context_new))) * context.detach()).detach()

            return latents_out, context_out

        # if scaler is not None:
        #     inv_scale = 1./scaler.get_scale()
        #     grads = [p * inv_scale for p in grads]
        # context_grad = context.grad
        # grad_cond = latents.grad

        grad_cond = grads[0]
        context_grad = grads[1]

        context_grad = torch.nan_to_num(context_grad, posinf=0.0, neginf=0.0, nan=0.0)
        grad_cond = torch.nan_to_num(grad_cond, posinf=0.0, neginf=0.0, nan=0.0)#latents.grad
        # if scaler is not None:
        #     scaler.update()

    context_new = None


    if grad_cond is not None:
        
        # if mask is not None and edit_type != "geometry_stitch":
        if mask is not None:
            mask_inpaint = reshape_attention_mask(mask[None, None].type_as(latents), in_mat_shape = latents[-1:].shape)       

            latents = torch.cat([latents[:-1], latents[-1:].detach() - 2.0 * mask_inpaint[-1:] * step_size * grad_cond[-1:]], 0)
            latents = torch.cat([latents[:-1], latents[-1:].detach() - (1.0 - mask_inpaint[-1:]) * step_size * grad_cond[-1:]], 0)

        # else:
            # print("latent grads")
        # print("Running latent grad", grad_cond[:-1].min(), grad_cond[:-1].max())
        # print("Running latent grad", grad_cond[-1:].min(), grad_cond[-1:].max())
        else:
            latents = torch.cat([latents[:-1], latents[-1:].detach() - step_size * grad_cond[-1:]], 0)

        if context is not None:

            # print("Running context grad", context_grad[:-1].min(), context_grad[:-1].max())
            # print("Running context grad", context_grad[-1:].min(), context_grad[-1:].max())

            context_new = torch.cat([context[:-1], context[-1:].detach() - step_size * context_grad[-1:]], 0)

            # context_new = torch.cat([context[:1], context[1:2].detach().clone() - step_size * context_grad[1:2], context[2:]], 0)
                
        
        
    else:
        print("no grad found")
    return latents, context_new

