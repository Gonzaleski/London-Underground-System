import pandas as pd
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from clrsPython import dijkstra, AdjacencyListGraph

# Load the data from the Excel file
df = pd.read_excel('London Underground data.xlsx', names=['Line', 'Source', 'Destination', 'Duration (minutes)'])
df = df.dropna(subset=['Destination'])  # Remove rows without destinations

stations = pd.concat([df['Source'], df['Destination']]).unique()
station_indices = {station: idx for idx, station in enumerate(stations)}

class TubeScreen(Screen):
    def open_source_dialog(self):
        content = GridLayout(cols=1, spacing=10, size_hint_y=None)
        content.bind(minimum_height=content.setter('height'))
        
        scroll_view = ScrollView(size_hint=(1, 1))
        for station in stations:
            btn = Button(text=station, size_hint_y=None, height=40)
            btn.bind(on_release=lambda btn: self.select_source(btn.text))
            content.add_widget(btn)
        scroll_view.add_widget(content)
        
        popup_content = GridLayout(cols=1)
        popup_content.add_widget(scroll_view)
        self.source_popup = Popup(title="Select Source Station", content=popup_content, size_hint=(0.8, 0.8))
        self.source_popup.open()

    def open_destination_dialog(self):
        content = GridLayout(cols=1, spacing=10, size_hint_y=None)
        content.bind(minimum_height=content.setter('height'))
        
        scroll_view = ScrollView(size_hint=(1, 1))
        for station in stations:
            btn = Button(text=station, size_hint_y=None, height=40)
            btn.bind(on_release=lambda btn: self.select_destination(btn.text))
            content.add_widget(btn)
        scroll_view.add_widget(content)
        
        popup_content = GridLayout(cols=1)
        popup_content.add_widget(scroll_view)
        self.destination_popup = Popup(title="Select Destination Station", content=popup_content, size_hint=(0.8, 0.8))
        self.destination_popup.open()

    def select_source(self, station_name):
        self.ids.source_button.text = station_name
        self.source_popup.dismiss()

    def select_destination(self, station_name):
        self.ids.destination_button.text = station_name
        self.destination_popup.dismiss()

    def calculate_path(self):
        source = self.ids.source_button.text
        destination = self.ids.destination_button.text
        option = self.ids.option_spinner.text
        
        if source == "Select Source" or destination == "Select Destination":
            self.ids.result_label.text = "Please select both source and destination."
            return
        
        source_idx = station_indices.get(source)
        destination_idx = station_indices.get(destination)
        
        if option == "Minutes":
            duration = self.find_shortest_path(source_idx, destination_idx, weight="Duration (minutes)")
            self.ids.result_label.text = f"Time taken from {source} to {destination}: {duration} minutes"
        elif option == "Stops":
            stops = self.find_shortest_path(source_idx, destination_idx, weight="Stops")
            self.ids.result_label.text = f"Number of stops from {source} to {destination}: {stops}"

    def find_shortest_path(self, source_idx, destination_idx, weight):
        # Create a graph representation with stations and durations
        graph = AdjacencyListGraph(len(stations), True, True)
        
        # Track edges that have already been added to avoid duplicates
        added_edges = set()
        
        for _, row in df.iterrows():
            source = station_indices[row['Source']]
            destination = station_indices[row['Destination']]
            duration = row['Duration (minutes)']
            
            # Check if the edge has already been added
            edge = (min(source, destination), max(source, destination))  # Ensure undirected consistency
            if edge not in added_edges:
                graph.insert_edge(source, destination, duration)
                added_edges.add(edge)
        
        # Run Dijkstra's algorithm to get shortest distances
        distances, predecessors = dijkstra(graph, source_idx)
        
        if weight == "Duration (minutes)":
            # Return shortest path duration in minutes to the destination
            return distances[destination_idx]
        elif weight == "Stops":
            # Calculate the number of stops based on the path length from source to destination
            stops = 0
            current = destination_idx
            while predecessors[current] is not None:
                current = predecessors[current]
                stops += 1
            return stops


class TubeApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(TubeScreen(name='tube_screen'))
        return sm

if __name__ == "__main__":
    TubeApp().run()
