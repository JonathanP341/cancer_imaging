import torch

def train_loop(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module, 
              optimizer: torch.optim.Optimizer, 
              device: torch.device) -> tuple[float, float]:
    """
    This will train a pytorch model for a single epoch

    Will run the epoch then return a tuple containing the training loss and accuracy
    """
    #Putting model in train mode
    model.train()

    #Setting up train loss
    train_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        #Moving X and y to the right device
        X, y = X.to(device), y.to(device)

        #Forward pass
        y_pred = model(X)

        #Calculate the loss and accuracy
        loss = loss_fn(y_pred, y)
        train_loss += loss

        #Other steps
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    #Adjusting the metrics
    train_loss /= len(dataloader)

    return train_loss.item()

def test_loop(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,  
              device: torch.device) -> tuple[float, float]:
    """
    This will test a pytorch model for a single epoch

    Will run the epoch then return a tuple containing the training loss and accuracy
    """
    #Putting model in evaluation mode
    model.eval()

    #Creating the loss & acc
    test_loss = 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            #Moving X and y to the right device
            X, y = X.to(device), y.to(device)

            #Forward pass
            y_pred = model(X)

            #Calculating loss and accuracy
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

    #Adjusting the metrics
    test_loss /= len(dataloader)

    return test_loss

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader,
          valid_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module, 
          optimizer: torch.optim.Optimizer, 
          epochs: int,
          device: torch.device) -> dict[str, list]:
    #Creating the output dictionary
    results = {"train_loss": [], "test_loss": [], "valid_loss": []}

    #Running the model
    for epoch in range(epochs):
        #Training and testing the loop
        train_loss = train_loop(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device)
        valid_loss = test_loop(model=model, dataloader=valid_dataloader, loss_fn=loss_fn, device=device)

        #Adding the values to the dictionary
        results["train_loss"].append(train_loss)
        results["valid_loss"].append(valid_loss)

        #Printing out the results
        print(f"--------------------Epoch {epoch + 1}--------------------")
        print(f"Train Loss: {train_loss:.4f} | Test Loss: {valid_loss:.4f}")

    #Checking with a final test set to see how the data learned overall on a new set of data
    test_loss = test_loop(model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device)
    results["test_loss"].append(test_loss)
    print(f"Test Loss: {test_loss:.4f}")

    #Returning the results
    return results