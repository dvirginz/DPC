import numpy as np
import os

from visualization.mesh_visualization_utils import (
    create_colormap,
)
from visualization.mesh_container import MeshContainer
from xvfbwrapper import Xvfb
from utils.tensor_utils import to_numpy

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from matplotlib import pyplot as plt



class MeshVisualizer(object):
    """Visualization class for meshes."""

    def __init__(self, dataset="FAUST", display_up=False):
        """
        The mesh visualization utility class

        Args:
            dataset (str, optional): Name of the dataset, e.g. will effect view angle . Defaults to 'FAUST'.
        """
        self.dataset = dataset
        self.scale = 0.8 if 'faust' in dataset else 0.2
        self.set_view_properties_by_dataset()
        if not display_up:
            vdisplay = Xvfb()
            vdisplay.start()

    def visualize_pcs(
        self,
        pcs,
        save_path="output/tmp/P_mesh_pair.png",
        write_html=True,
        horiz_space=0,
    ):
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        for idx in range(len(pcs)):
            if not type(pcs[idx]).__module__ == np.__name__:
                pcs[idx] = pcs[idx].detach().cpu().numpy() 
                
        fig = make_subplots(
            rows=1,
            cols=len(pcs),
            shared_yaxes=True,
            shared_xaxes=True,
            specs=[[{"type": "scatter3d"} for i in range(len(pcs))]],
            horizontal_spacing=horiz_space,
        )
        for idx,pc in enumerate(pcs):
            fig.add_trace(
                self.plotly_mesh(
                    MeshContainer(vert=pc),
                    color_map_vis=create_colormap(pc),
                    mesh_or_pc="pc",
                ),
                row=1,
                col=idx+1,
            )

        camera = dict(up=dict(x=0, y=1.0, z=0), eye=dict(x=-0.0, y=-0.0, z=5))

        with fig.batch_update():
            for idx in range(len(pcs)):
            # scale = np.max((np.max(np.abs(target_mesh.vert),0),np.max(np.abs(source_mesh.vert),0)),0)
                self.set_fig_settings(scene=eval(f"fig.layout.scene{idx+1 if idx>0 else ''}"), fig=fig, camera=camera,scale=self.scale)

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            autosize=False, width=600*len(pcs), height=900,
        )

        fig.write_image(save_path,)
        if write_html:
            fig.write_html(save_path + ".html")
        return fig

    def visualize_pc_pair_same_fig(
        self,
        pcs,
        save_path="output/tmp/P_pc_pair_same_fig.png",
        write_html=True,
        horiz_space=0,
    ):
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        for idx in range(len(pcs)):
            if not type(pcs[idx]).__module__ == np.__name__:
                pcs[idx] = pcs[idx].detach().cpu().numpy() 
                
        fig = make_subplots(
            rows=1,
            cols=1,
            shared_yaxes=True,
            shared_xaxes=True,
            specs=[[{"type": "scatter3d"} for i in range(1)]],
            horizontal_spacing=horiz_space,
        )
        green,blue = (0,0.45,0),(0,0,1)
        for idx,pc in enumerate(pcs):
            color = green if idx == 0 else blue
            fig.add_trace(
                self.plotly_mesh(
                    MeshContainer(vert=pc),
                    color_map_vis=np.stack([color for i in range(pc.shape[0])]),
                    mesh_or_pc="pc",
                ),
                row=1,
                col=1,
            )

        camera = dict(up=dict(x=0, y=1.0, z=0), eye=dict(x=-0.0, y=-0.0, z=5))

        with fig.batch_update():
            for idx in range(1):
            # scale = np.max((np.max(np.abs(target_mesh.vert),0),np.max(np.abs(source_mesh.vert),0)),0)
                self.set_fig_settings(scene=eval(f"fig.layout.scene{idx+1 if idx>0 else ''}"), fig=fig, camera=camera,scale=self.scale)

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            autosize=False, width=600, height=900,
        )

        fig.write_image(save_path,)
        if write_html:
            fig.write_html(save_path + ".html")
        return fig

    def visualize_pcs_same_fig(
            self,
            pcs,
            pcs_color=[(0, 0.5, 0), (0, 0, 0.5), (0.5, 0, 0)],
            save_path="output/tmp/pc.png",
            use_text=True,
            write_image=True,
            write_html=True,
            horiz_space=0,
    ):
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        for idx in range(len(pcs)):
            if not type(pcs[idx]).__module__ == np.__name__:
                pcs[idx] = pcs[idx].detach().cpu().numpy()

        fig = make_subplots(
            rows=1,
            cols=1,
            shared_yaxes=True,
            shared_xaxes=True,
            specs=[[{"type": "scatter3d"} for i in range(1)]],
            horizontal_spacing=horiz_space,
        )

        for idx, pc in enumerate(pcs):
            fig.add_trace(
                self.plotly_mesh(
                    MeshContainer(vert=pc),
                    color_map_vis=np.stack([pcs_color[idx] for i in range(pc.shape[0])]),
                    mesh_or_pc="pc", use_text=use_text,
                ),
                row=1,
                col=1,
            )

        camera = dict(up=dict(x=0, y=1.0, z=0), eye=dict(x=-0.0, y=-0.0, z=5))

        with fig.batch_update():
            for idx in range(1):
                # scale = np.max((np.max(np.abs(target_mesh.vert),0),np.max(np.abs(source_mesh.vert),0)),0)
                self.set_fig_settings(scene=eval(f"fig.layout.scene{idx + 1 if idx > 0 else ''}"), fig=fig,
                                      camera=camera, scale=self.scale)

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            autosize=False, width=600, height=900,
        )

        if write_image:
            fig.write_image(save_path, )

        if write_html:
            fig.write_html(save_path + ".html")

        return fig

    def visualize_mesh_pair(
        self,
        source_mesh,
        target_mesh,
        corr_map,
        title=None,
        color_map=None,
        is_show=False,
        save_path="output/tmp/P_mesh_pair.png",
        mesh_or_pc='mesh',
        fwd_or_bac='fwd',
        horiz_space=0.1,
        grayed_indices=None,
        target_grayed=None,
        rotate_pc_for_vis=False,
        rotate_pc_angles=None,
        write_image=True,
        write_html=True,
        vis_mitsuba=False,
    ):
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if fwd_or_bac == 'fwd':  # forward mapping
            color_map_vis = color_map if color_map is not None else create_colormap(target_mesh.vert)
        else:  # backward mapping
            color_map_vis = color_map if color_map is not None else create_colormap(source_mesh.vert)

        if grayed_indices is not None:
            color_map_vis[grayed_indices] = 0

        mapping_colors = color_map_vis[to_numpy(corr_map)]
        if target_grayed is not None:
            mapping_colors[target_grayed] = 0

        if fwd_or_bac == 'fwd':  # in forward mapping, the target is colored by the ground truth colors, and the source to target mapping is colored accordingly
            source_colors, target_colors = mapping_colors, color_map_vis
        else:  # it backward mapping, the target "draws" the colors from the source
            source_colors, target_colors = color_map_vis, mapping_colors

        if not write_image and not write_html:
            return None, source_colors, target_colors

        fig = make_subplots(
            rows=1,
            cols=2,
            shared_yaxes=True,
            shared_xaxes=True,
            specs=[[{"type": "mesh3d"}, {"type": "mesh3d"}]],
            horizontal_spacing=horiz_space,
        )
        fig.add_trace(
            self.plotly_mesh(
                source_mesh,
                np.concatenate(
                    [source_colors, np.ones(source_colors.shape[0])[:, None]], axis=1
                ),
                mesh_or_pc=mesh_or_pc,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            self.plotly_mesh(
                target_mesh,
                np.concatenate(
                    [target_colors, np.ones(target_colors.shape[0])[:, None]],
                    axis=1,
                ),
                mesh_or_pc=mesh_or_pc,
            ),
            row=1,
            col=2,
        )

        camera = dict(up=dict(x=0, y=1.0, z=0), eye=dict(x=-0.0, y=-0.0, z=5))

        with fig.batch_update():
            
            # scale = np.max((np.max(np.abs(target_mesh.vert),0),np.max(np.abs(source_mesh.vert),0)),0)
            self.set_fig_settings(scene=fig.layout.scene, fig=fig, camera=camera,scale=self.scale)
            self.set_fig_settings(scene=fig.layout.scene2, fig=fig, camera=camera,scale=self.scale)

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            autosize=False, width=1200, height=900,
        )

        if write_image:
            fig.write_image(save_path,)

        if write_html:
            fig.write_html(save_path + ".html")

        return fig, source_colors, target_colors

    @staticmethod
    def set_fig_settings(scene, fig, camera, width=600, height=600,scale=False):
        fig.layout.update(autosize=False,width=width, height=height, margin=dict(l=0, r=0, t=0, b=0))

        scene.camera = camera
        scene.aspectmode = "data"
        scene.xaxis.visible = False
        scene.yaxis.visible = False
        scene.zaxis.visible = False
        # if(scale is not False):
        #     scene.xaxis.range = [-scale[0],scale[0]]
        #     scene.yaxis.range = [-scale,scale]
        #     scene.zaxis.range = [-scale,scale]

    def plotly_normals(self, points, normals):
        return go.Cone(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],  # i, j and k give the vertices of triangles
            u=normals[:, 0],
            v=normals[:, 1],
            w=normals[:, 2],
            showlegend=False,
            showscale=False,
            hoverinfo="text",
            text=[str(idx) for idx in range(6890)],
            sizemode="scaled",
            sizeref=2,
            anchor="tip",
            # lighting=dict(ambient=0.4, diffuse=0.6, roughness=0.9,),
            # lightposition=dict(z=5000),
        )

    def plotly_mesh(self, source_mesh, color_map_vis, mesh_or_pc="mesh", use_text=True):
        if use_text:
            hoverinfo = "text"
            text = [str(idx) for idx in range(6890)]
        else:
            hoverinfo = None
            text = None

        if mesh_or_pc == "mesh":
            return go.Mesh3d(
                x=source_mesh.vert[:, 0],
                y=source_mesh.vert[:, 1],
                z=source_mesh.vert[:, 2],  # i, j and k give the vertices of triangles
                i=source_mesh.face[:, 0],
                j=source_mesh.face[:, 1],
                k=source_mesh.face[:, 2],
                vertexcolor=color_map_vis,
                showlegend=False,
                hoverinfo=hoverinfo,
                text=text,
                lighting=dict(ambient=0.4, diffuse=0.6, roughness=0.9,),
                lightposition=dict(z=5000),
            )
        else:
            try:
                color = ['rgb(' + str(int(c[0]*255)) + ',' + str(int(255*c[1])) + ',' + str(int(255*c[2])) + ')' for c in color_map_vis]
            except:
                color = ['rgb(' + str(0) + ',' + str(0) + ',' + str(0) + ')' for c in color_map_vis]

            return go.Scatter3d(
                x=source_mesh.vert[:, 0],
                y=source_mesh.vert[:, 1],
                z=source_mesh.vert[:, 2],  # i, j and k give the vertices of triangles
                mode='markers',
                marker=dict(
                    size=6,
                    color=color,                # set color to an array/list of desired values
                ),
                showlegend=False,
                hoverinfo=hoverinfo,
                text=text,
            )

    def scale_by_dataset(self, mesh):
        """For specific datasets we scale the shape for better visualization."""
        if self.dataset == "TOSCA":
            mesh.vert = mesh.vert * 100
        return mesh

    def set_view_properties_by_dataset(self):
        pass
        # if(self.dataset == 'FAUST'):
        #     mlab.view(0, 0)
        #     self.displacment = 1
        # if(self.dataset == 'TOSCA'):
        #     mlab.view(50, 90)

    @staticmethod
    def visualize_scalar_vector_on_shape_static(verts,face,scalar_map,display_up=False,extra_text="",normals=None,max_scalar=None,):
        return MeshVisualizer(dataset='faust',display_up=display_up).visualize_scalar_vector_on_shape(
            source_mesh=MeshContainer(verts, face),
            scalar_represent_color_vector=scalar_map,
            save_path=f"output/tmp/{extra_text}_.png",
            mesh_or_pc='mesh' if face is not None else 'pc',
            normals=normals,
            max_scalar=None,
        )

    def visualize_scalar_vector_on_shape(
        self,
        source_mesh,
        scalar_represent_color_vector,
        save_path="output/tmp/vis_scalar_on_shape.png",
        max_scalar=None,
        write_html=True,
        mesh_or_pc='mesh',
        normals=None
    ):
        if not type(scalar_represent_color_vector).__module__ == np.__name__:
            scalar_represent_color_vector = (
                scalar_represent_color_vector.cpu().detach().numpy()
            )
        if normals is not None and not type(normals).__module__ == np.__name__:
            normals = normals.cpu().detach().numpy()

        scalar_represent_color_vector = scalar_represent_color_vector / (
            max_scalar or scalar_represent_color_vector.max()
        )
        Viridis = plt.get_cmap("viridis")
        colors = [Viridis(scalar) for scalar in scalar_represent_color_vector]
        if(normals is None):
            fig = go.Figure(
                data=[self.plotly_mesh(source_mesh=source_mesh, color_map_vis=colors,mesh_or_pc=mesh_or_pc),]
            )
        else:
           fig = go.Figure(data=[self.plotly_normals(points=source_mesh.vert,normals=normals)])
        camera = dict(up=dict(x=0, y=1.0, z=0), eye=dict(x=-0.0, y=-0.0, z=5))

        with fig.batch_update():
            self.set_fig_settings(scene=fig.layout.scene, fig=fig, camera=camera)

        fig.update_layout(
            autosize=False, width=400, height=600,
        )
        if write_html:
            fig.write_html(save_path + ".html")
        fig.write_image(save_path)

        return fig

    @staticmethod
    def get_pcs_range(pcs):
        min_vals_list = [None] * len(pcs)
        max_vals_list = [None] * len(pcs)
        for idx in range(len(pcs)):
            if not type(pcs[idx]).__module__ == np.__name__:
                pcs[idx] = pcs[idx].detach().cpu().numpy()

            min_vals_list[idx] = np.min(pcs[idx], axis=0)
            max_vals_list[idx] = np.max(pcs[idx], axis=0)

        pcs_min_vals = np.min(np.vstack(min_vals_list), axis=0)
        pcs_max_vals = np.max(np.vstack(max_vals_list), axis=0)
        pcs_range = np.vstack([pcs_min_vals, pcs_max_vals])

        return pcs_range

    @staticmethod
    def util_vis_from_paths(
        shape1_path, shape2_path, corr_map, save_path, dataset="FAUST",
    ):
        mesh_container1 = MeshContainer().load_from_file(shape1_path)
        mesh_container2 = MeshContainer().load_from_file(shape2_path)

        return MeshVisualizer(dataset).visualize_mesh_pair(
            source_mesh=mesh_container1,
            target_mesh=mesh_container2,
            corr_map=corr_map,  # corr,#np.arange(6270),
            save_path=save_path,
        )

    @staticmethod
    def visualize_correspondence_over_time(
        figs, samples_per_row=2, save_path="output/tmp/over_time.png"
    ):
        """Takes several pairs of shape correspondence and plot one beneaf the other."""
        super_fig = make_subplots(
            rows=len(figs),
            cols=samples_per_row,
            shared_yaxes=True,
            shared_xaxes=True,
            specs=[[{"type": "mesh3d"}] * samples_per_row] * len(figs),
            horizontal_spacing=-0.75,
            vertical_spacing=-0.05,
        )
        for idx, fig in enumerate(figs):
            # super_fig.add_trace(fig.data[1], row=idx + 1, col=1)
            # super_fig.add_trace(fig.data[2], row=idx + 1, col=2)
            for data_idx, data in enumerate(fig.data):
                super_fig.add_trace(data, row=idx + 1, col=data_idx + 1)
        
        camera = dict(up=dict(x=0, y=1.0, z=0), eye=dict(x=-0.0, y=-0.0, z=5))

        with super_fig.batch_update():
            for i in range(1, (samples_per_row * len(figs)) + 1):
                scene = getattr(super_fig.layout, "scene" + ("" if i == 0 else str(i)))
                MeshVisualizer.set_fig_settings(
                    scene=scene,
                    fig=super_fig,
                    camera=camera,
                    width=600 * len(figs),
                    height=600 * len(figs),
                )

        super_fig.write_image(save_path)
        super_fig.write_html(save_path + ".html")
        return super_fig

    @staticmethod
    def plot_3d_point_cloud(pc, show=True, show_axis=True, in_u_sphere=True, marker='.', c='b', s=8, alpha=.8, figsize=(5, 5),
                            elev=90, azim=-90, miv=None, mav=None, squeeze=0.7, axis=None, title=None, *args, **kwargs):
        import matplotlib
        matplotlib.use('TKAgg', force=True)

        x, y, z = (pc[:, 0], pc[:, 1], pc[:, 2])

        if axis is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = axis
            fig = axis

        if title is not None:
            plt.title(title)

        sc = ax.scatter(x, y, z, marker=marker, c=c, s=s, alpha=alpha, *args, **kwargs)
        ax.view_init(elev=elev, azim=azim)

        if in_u_sphere:
            ax.set_xlim3d(-1, 1)
            ax.set_ylim3d(-1, 1)
            ax.set_zlim3d(-1, 1)
            miv = -1
            mav = 1
        else:
            if miv is None:
                miv = squeeze * np.min(
                    [np.min(x), np.min(y), np.min(z)])  # Multiply with 'squeeze' to squeeze free-space.
            if mav is None:
                mav = squeeze * np.max([np.max(x), np.max(y), np.max(z)])
            ax.set_xlim(miv, mav)
            ax.set_ylim(miv, mav)
            ax.set_zlim(miv, mav)
            plt.tight_layout()

        if not show_axis:
            plt.axis('off')

        if 'c' in kwargs:
            plt.colorbar(sc)

        if show:
            plt.show()

        return fig, miv, mav


