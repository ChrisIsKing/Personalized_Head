import time,logging
import pandas as pd
from torch.utils.data import Dataset
from datasets import load_dataset
import pickle
import random

random.seed(4)

intent= [
    "restaurant_reviews",
    "nutrition_info",
    "account_blocked",
    "oil_change_how",
    "time",
    "weather",
    "redeem_rewards",
    "interest_rate",
    "gas_type",
    "accept_reservations",
    "smart_home",
    "user_name",
    "report_lost_card",
    "repeat",
    "whisper_mode",
    "what_are_your_hobbies",
    "order",
    "jump_start",
    "schedule_meeting",
    "meeting_schedule",
    "freeze_account",
    "what_song",
    "meaning_of_life",
    "restaurant_reservation",
    "traffic",
    "make_call",
    "text",
    "bill_balance",
    "improve_credit_score",
    "change_language",
    "no",
    "measurement_conversion",
    "timer",
    "flip_coin",
    "do_you_have_pets",
    "balance",
    "tell_joke",
    "last_maintenance",
    "exchange_rate",
    "uber",
    "car_rental",
    "credit_limit",
    "shopping_list",
    "expiration_date",
    "routing",
    "meal_suggestion",
    "tire_change",
    "todo_list",
    "card_declined",
    "rewards_balance",
    "change_accent",
    "vaccines",
    "reminder_update",
    "food_last",
    "change_ai_name",
    "bill_due",
    "who_do_you_work_for",
    "share_location",
    "international_visa",
    "calendar",
    "translate",
    "carry_on",
    "book_flight",
    "insurance_change",
    "todo_list_update",
    "timezone",
    "cancel_reservation",
    "transactions",
    "credit_score",
    "report_fraud",
    "spending_history",
    "directions",
    "spelling",
    "insurance",
    "what_is_your_name",
    "reminder",
    "where_are_you_from",
    "distance",
    "payday",
    "flight_status",
    "find_phone",
    "greeting",
    "alarm",
    "order_status",
    "confirm_reservation",
    "cook_time",
    "damaged_card",
    "reset_settings",
    "pin_change",
    "replacement_card_duration",
    "new_card",
    "roll_dice",
    "income",
    "taxes",
    "date",
    "who_made_you",
    "pto_request",
    "tire_pressure",
    "how_old_are_you",
    "rollover_401k",
    "pto_request_status",
    "how_busy",
    "application_status",
    "recipe",
    "calendar_update",
    "play_music",
    "yes",
    "direct_deposit",
    "credit_limit_change",
    "gas",
    "pay_bill",
    "ingredients_list",
    "lost_luggage",
    "goodbye",
    "what_can_i_ask_you",
    "book_hotel",
    "are_you_a_bot",
    "next_song",
    "change_speed",
    "plug_type",
    "maybe",
    "w2",
    "oil_change_when",
    "thank_you",
    "shopping_list_update",
    "pto_balance",
    "order_checks",
    "travel_alert",
    "fun_fact",
    "sync_device",
    "schedule_maintenance",
    "apr",
    "transfer",
    "ingredient_substitution",
    "calories",
    "current_location",
    "international_fees",
    "calculator",
    "definition",
    "next_holiday",
    "update_playlist",
    "mpg",
    "min_payment",
    "change_user_name",
    "restaurant_suggestion",
    "travel_notification",
    "cancel",
    "pto_used",
    "travel_suggestion",
    "change_volume"
  ]

# dataset = load_dataset("clinc_oos","plus")
with open('clinc/train_data_clinc_wo_oos.pkl', 'rb') as file:
    train = pickle.load(file)
    print(train[0].to_plain_string())
    print(train[1].get_labels("clinc_data")[0].value)


def create_dataset_clinc():
    train_text,test_text,dev_text=[],[],[]
    train_class,test_class,dev_class=[],[],[]
    count=0

    for i in range(len(dataset['train']["intent"])):
        if dataset['train']["intent"][i]==42:
            if count<100:
                train_text.append(dataset['train']["text"][i])
                train_class.append(dataset['train']["intent"][i])
                count+=1
            else:
                continue
        else:
            train_text.append(dataset['train']["text"][i])
            train_class.append(dataset['train']["intent"][i])
    count=0
    for i in range(len(dataset['test']["intent"])):
        if dataset['test']["intent"][i]==42:
            if count<30:
                test_text.append(dataset['test']["text"][i])
                test_class.append(dataset['test']["intent"][i])
                count+=1
            else:
                continue
        else:
            test_text.append(dataset['test']["text"][i])
            test_class.append(dataset['test']["intent"][i])
    count=0
    for i in range(len(dataset['validation']["intent"])):
        if dataset['validation']["intent"][i]==42:
            if count<20:
                dev_text.append(dataset['validation']["text"][i])
                dev_class.append(dataset['validation']["intent"][i])
                count+=1
            else:
                continue
        else:
            dev_text.append(dataset['validation']["text"][i])
            dev_class.append(dataset['validation']["intent"][i])
    return  train_text,train_class,test_text,test_class,dev_text,dev_class


def format_data(raw, tokenizer, neg_sample=True):
    texts = []
    labels = []
    label_names = []
    for i in range(len(raw)):
        label = raw[i].get_labels("clinc_data")[0].value
        original_text = raw[i].to_plain_string()
        text = f"{label} {tokenizer.sep_token} {original_text}" 
        texts.append(text)
        labels.append(1)
        label_names.append(label)
        if neg_sample:
            other_labels = random.sample([other_label for other_label in intent if other_label != label], 2)
            for l in other_labels:
                text = f"{l} {tokenizer.sep_token} {original_text}" 
                texts.append(text)
                labels.append(0)
                label_names.append(l)
    return texts, labels, label_names

def format_infer_data(raw, tokenizer):
    texts = []
    labels = []
    label_names = []
    for i in range(len(raw)):
        label = raw[i].get_labels("clinc_data")[0].value
        original_text = raw[i].to_plain_string()
        text = f"{label} {tokenizer.sep_token} {original_text}" 
        texts.append(text)
        labels.append(1)
        label_names.append(label)
        other_labels = [other_label for other_label in intent if other_label != label]
        for l in other_labels:
            text = f"{l} {tokenizer.sep_token} {original_text}" 
            texts.append(text)
            labels.append(0)
            label_names.append(l)
    return texts, labels, label_names

def get_clinc_dataset(tokenizer):
    # train_text,train_class,test_text,test_class,dev_text,dev_class=create_dataset_clinc()
    with open('clinc/train_data_clinc_wo_oos.pkl', 'rb') as file:
        train = pickle.load(file)
    with open('clinc/test_data_clinc_wo_oos.pkl', 'rb') as file:
        test = pickle.load(file)
    with open('clinc/dev_data_clinc_wo_oos.pkl', 'rb') as file:
        dev = pickle.load(file)
    train_text, train_class, train_labels = format_data(train, tokenizer, neg_sample=True)
    test_text, test_class, test_labels = format_infer_data(test, tokenizer)
    dev_text, dev_class, dev_labels = format_data(dev, tokenizer, neg_sample=False)
    return ClincDataset(train_text, train_class, tokenizer, train_labels), ClincDataset(test_text, test_class, tokenizer, test_labels), ClincDataset(dev_text, dev_class, tokenizer, dev_labels)



class ClincDataset(Dataset):
    def __init__(self, text, label, tokenizer, name):
        self.text = text
        self.tokenizer = tokenizer
        self.label = label
        self.name = name
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        item = self.tokenizer(self.text[idx], max_length=80, truncation=True, padding="max_length")
        item["labels"] = self.label[idx]
        item["name"] = self.name[idx]
        return item